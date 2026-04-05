import torch
import psutil
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, ProgressBar, Static

from vector_field.trainer import Trainer


class MetricsPanel(Static):
    epoch: int = reactive(0)
    loss: float = reactive(float("inf"))
    cpu: float = reactive(0.0)
    ram: float = reactive(0.0)

    def render(self) -> str:
        return (
            f"[bold]Metrics[/bold]\n\n"
            f"Epoch  [dim]│[/dim] {self.epoch}\n"
            f"Loss   [dim]│[/dim] {self.loss:.6f}\n"
            f"CPU    [dim]│[/dim] {self.cpu:.1f}%\n"
            f"RAM    [dim]│[/dim] {self.ram:.1f} GB\n"
        )


class LossPlot(Static):
    history: list[float] = reactive([], recompose=True)

    _BLOCKS = "▁▂▃▄▅▆▇█"

    def render(self) -> str:
        if not self.history:
            return "[dim]Waiting for data...[/dim]"

        lo, hi = min(self.history), max(self.history)
        span = hi - lo + 1e-9
        sparkline = "".join(
            self._BLOCKS[int((v - lo) / span * (len(self._BLOCKS) - 1))]
            for v in self.history
        )
        return (
            f"[bold]Loss History[/bold]\n\n"
            f"{sparkline}\n\n"
            f"[dim]min {lo:.4f}  max {hi:.4f}[/dim]"
        )


class TrainingTUI(App):
    CSS = """
    Screen { layout: vertical; }

    #panels {
        height: 1fr;
        layout: horizontal;
    }

    MetricsPanel {
        width: 32;
        border: round $primary;
        padding: 1 2;
    }

    LossPlot {
        width: 1fr;
        border: round $primary;
        padding: 1 2;
    }

    ProgressBar { margin: 1 2; }
    """

    def __init__(
        self,
        trainer: Trainer,
        num_epochs: int,
        device: torch.device,
        lr: float = 1e-3,
        **train_kwargs,
    ):
        super().__init__()
        self._trainer = trainer
        self._num_epochs = num_epochs
        self._device = device
        self._lr = lr
        self._train_kwargs = train_kwargs

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="panels"):
            yield MetricsPanel()
            yield LossPlot()
        yield ProgressBar(total=self._num_epochs, show_eta=True)
        yield Footer()

    def on_mount(self) -> None:
        # run_worker with thread=True offloads blocking work to a thread pool
        # while keeping Textual's async event loop unblocked
        self.run_worker(self._training_worker, thread=True)

    def _training_worker(self) -> None:
        def on_step(epoch: int, loss: float) -> None:
            # call_from_thread queues the callable onto the event loop thread,
            # since Textual's DOM is not thread-safe
            self.call_from_thread(self._update_ui, epoch, loss)

        self._trainer.train(
            num_epochs=self._num_epochs,
            device=self._device,
            lr=self._lr,
            on_step=on_step,
            **self._train_kwargs,
        )

    def _update_ui(self, epoch: int, loss: float) -> None:
        metrics = self.query_one(MetricsPanel)
        metrics.epoch = epoch + 1
        metrics.loss = loss
        metrics.cpu = psutil.cpu_percent(interval=None)
        metrics.ram = psutil.virtual_memory().used / 1e9

        plot = self.query_one(LossPlot)
        plot.history = plot.history + [loss]

        self.query_one(ProgressBar).advance(1)
