# AI_GENERATE_START
import asyncio
import json
import time
from typing import Any, Optional


class Step:
    """单个步骤事件"""

    def __init__(self, step: int, phase: str, title: str, code: str,
                 node_id: Optional[str] = None,
                 node_status: Optional[str] = None,
                 input_data: Optional[Any] = None,
                 output_data: Optional[Any] = None,
                 explanation: Optional[str] = None,
                 component: Optional[str] = None,
                 elapsed_ms: int = 0,
                 all_nodes_status: Optional[dict] = None):
        self.step = step
        self.phase = phase
        self.title = title
        self.code = code
        self.node_id = node_id
        self.node_status = node_status
        self.input_data = input_data
        self.output_data = output_data
        self.explanation = explanation
        self.component = component
        self.elapsed_ms = elapsed_ms
        self.all_nodes_status = all_nodes_status or {}

    def to_dict(self) -> dict:
        return {
            'step': self.step,
            'phase': self.phase,
            'title': self.title,
            'code': self.code,
            'node_id': self.node_id,
            'node_status': self.node_status,
            'input_data': _truncate(self.input_data),
            'output_data': _truncate(self.output_data),
            'explanation': self.explanation,
            'component': self.component,
            'elapsed_ms': self.elapsed_ms,
            'all_nodes_status': self.all_nodes_status,
        }


def _truncate(data: Any, max_len: int = 8000) -> Any:
    """截断过长的数据"""
    if data is None:
        return None
    s = json.dumps(data, ensure_ascii=False, default=str)
    if len(s) > max_len:
        # 直接返回截断的字符串，不尝试解析回 JSON
        return s[:max_len] + '...'
    return data


class StepTracer:
    """步骤追踪器，记录每一步并通过队列推送给 SSE"""

    def __init__(self):
        self._step_counter = 0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._history: list[Step] = []
        # 节点状态追踪
        self._nodes_status: dict[str, str] = {}

    def trace(self, phase: str, title: str, code: str,
              node_id: Optional[str] = None,
              node_status: Optional[str] = None,
              input_data: Optional[Any] = None,
              output_data: Optional[Any] = None,
              explanation: Optional[str] = None,
              component: Optional[str] = None,
              elapsed_ms: int = 0):
        """记录一个步骤事件"""
        self._step_counter += 1

        # 更新节点状态
        if node_id and node_status:
            self._nodes_status[node_id] = node_status

        step = Step(
            step=self._step_counter,
            phase=phase,
            title=title,
            code=code,
            node_id=node_id,
            node_status=node_status,
            input_data=input_data,
            output_data=output_data,
            explanation=explanation,
            component=component,
            elapsed_ms=elapsed_ms,
            all_nodes_status=dict(self._nodes_status),
        )
        self._history.append(step)
        self._queue.put_nowait(step)

    def trace_with_timing(self, phase: str, title: str, code: str, **kwargs):
        """带计时的 trace 装饰，返回一个上下文管理器"""
        return _TimingContext(self, phase, title, code, **kwargs)

    async def get_events(self):
        """SSE 事件生成器"""
        while True:
            step = await self._queue.get()
            if step is None:
                # 结束信号
                yield 'data: {"done": true}\n\n'
                break
            data = json.dumps(step.to_dict(), ensure_ascii=False, default=str)
            yield f'data: {data}\n\n'

    def finish(self):
        """发送结束信号"""
        self._queue.put_nowait(None)

    def get_history(self) -> list[dict]:
        """获取所有历史步骤"""
        return [s.to_dict() for s in self._history]

    def reset(self):
        """重置追踪器"""
        self._step_counter = 0
        self._history.clear()
        self._nodes_status.clear()
        # 清空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class _TimingContext:
    """计时上下文管理器"""

    def __init__(self, tracer: StepTracer, phase: str, title: str, code: str, **kwargs):
        self._tracer = tracer
        self._phase = phase
        self._title = title
        self._code = code
        self._kwargs = kwargs
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = int((time.time() - self._start) * 1000)
        self._tracer.trace(
            self._phase, self._title, self._code,
            elapsed_ms=elapsed, **self._kwargs
        )

    # 支持设置输出数据
    def set_output(self, data: Any):
        self._kwargs['output_data'] = data
# AI_GENERATE_END
