from pathlib import Path


def test_optimizer_step_precedes_structural_post_backward():
    source = Path('prodig/trainer.py').read_text()
    begin = source.index('def train_steps')
    end = source.index('@torch.no_grad()', begin)
    body = source[begin:end]
    assert body.index('self._optimizer_step()') < body.index('self.strategy.step_post_backward(')
    assert 'packed=self.cfg.training.packed' in body
