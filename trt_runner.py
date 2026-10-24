"""Executa um motor TensorRT (.engine) no lugar do modelo PyTorch.

`TRTModule(engine_path, device)(imagem)` devolve os mesmos logits que
`modelo(imagem)` — então decodificação, métricas, CSV e fusão funcionam sem
mudança. Usado pelos `*/infer.py` com `--compress trt_fp16|trt_fp32 --trt-engine`.

Os buffers de entrada/saída são tensores CUDA do próprio PyTorch (sem pycuda).
Suporta a API do TensorRT 8.x (Jetson, JetPack 4.6) e do 10.x. Compatível com
Python 3.6.

O motor é compilado pra uma GPU específica: gerar na própria Jetson com
`build_trt_engines.sh` (a partir dos ONNX do `export_onnx.py`).
"""

from typing import List, Tuple

import torch


def _dtype_map(trt) -> dict:
    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }


class TRTModule:
    """Motor TensorRT com interface de modelo PyTorch (1 entrada, 1+ saídas)."""

    def __init__(self, engine_path, device: torch.device) -> None:
        import tensorrt as trt

        if device.type != "cuda":
            raise ValueError("TensorRT requer --device cuda.")
        self.device = device
        self.engine_path = str(engine_path)
        with open(self.engine_path, "rb") as f:
            self.engine_bytes = f.read()
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self.engine = self._runtime.deserialize_cuda_engine(self.engine_bytes)
        if self.engine is None:
            raise RuntimeError(
                "Não foi possível carregar {} — o motor foi gerado em outra GPU ou outra "
                "versão do TensorRT? Gere de novo na própria Jetson.".format(self.engine_path)
            )
        self.context = self.engine.create_execution_context()

        dmap = _dtype_map(trt)
        self._new_api = not hasattr(self.engine, "num_bindings")  # TensorRT 10+
        self.inputs = []  # type: List[Tuple[str, tuple, torch.dtype, int]]
        self.outputs = []  # type: List[Tuple[str, tuple, torch.dtype, int]]
        if self._new_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                spec = (name, tuple(self.engine.get_tensor_shape(name)),
                        dmap[self.engine.get_tensor_dtype(name)], i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append(spec)
                else:
                    self.outputs.append(spec)
        else:
            for i in range(self.engine.num_bindings):
                spec = (self.engine.get_binding_name(i), tuple(self.engine.get_binding_shape(i)),
                        dmap[self.engine.get_binding_dtype(i)], i)
                if self.engine.binding_is_input(i):
                    self.inputs.append(spec)
                else:
                    self.outputs.append(spec)
        if len(self.inputs) != 1:
            raise ValueError("Esperava 1 entrada no motor, achei {}.".format(len(self.inputs)))
        self._out_buffers = [
            torch.empty(shape, dtype=dtype, device=device) for _, shape, dtype, _ in self.outputs
        ]
        # Memória que o TensorRT aloca por fora do PyTorch (invisível pro
        # torch.cuda.max_memory_allocated): área de trabalho das ativações + pesos
        # (aproximados pelo tamanho do motor). Somada em bench.gpu_memory_mb.
        activations = getattr(self.engine, "device_memory_size_v2", None)
        if activations is None:
            activations = getattr(self.engine, "device_memory_size", 0)
        self.gpu_mem_mb = (int(activations) + len(self.engine_bytes)) / (1024 ** 2)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Roda o motor. A saída é um buffer reaproveitado entre chamadas — consumir
        (decodificar/copiar) antes da próxima chamada."""
        name, shape, dtype, idx = self.inputs[0]
        x = x.to(device=self.device, dtype=dtype).contiguous()
        if tuple(x.shape) != shape:
            raise ValueError(
                "Entrada com shape {} mas o motor espera {} (o motor tem batch/resolução "
                "fixos — usar --batch-size 1 e a mesma resolução do ONNX).".format(tuple(x.shape), shape)
            )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if self._new_api:
            self.context.set_tensor_address(name, x.data_ptr())
            for (out_name, _, _, _), buf in zip(self.outputs, self._out_buffers):
                self.context.set_tensor_address(out_name, buf.data_ptr())
            ok = self.context.execute_async_v3(stream)
        else:
            bindings = [0] * (len(self.inputs) + len(self.outputs))
            bindings[idx] = x.data_ptr()
            for (_, _, _, out_idx), buf in zip(self.outputs, self._out_buffers):
                bindings[out_idx] = buf.data_ptr()
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream)
        if not ok:
            raise RuntimeError("Falha ao executar o motor TensorRT.")
        return self._out_buffers[0]

    # Compatibilidade com chamadas de modelo PyTorch feitas pelos scripts
    def eval(self):
        return self

    def parameters(self):
        return iter(())
