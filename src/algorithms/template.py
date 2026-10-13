"""Copy into a paper method directory, then implement its two device stages."""

from torch import nn


class PaperDetector(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        # register_buffer(...) for persistent projections and online state.

    def prepare(self, a, b, *, phase, seed):
        """Return reference checks generated from PackedOperand Q+S.

        b stores logical B transposed, as [N,K]. Both operands are still clean.
        Reference tensors must be independent of subsequent operand mutation.
        """
        raise NotImplementedError("在此实现论文的编码和参考校验。")

    def verify(self, output, prepared, *, correct):
        """Return device tensors (status[6], residual, threshold).

        Persist only updates allowed by the method; do not convert to host
        scalars here. correct=False leaves output unchanged.
        """
        raise NotImplementedError("在此实现论文的阈值、判定和恢复。")
