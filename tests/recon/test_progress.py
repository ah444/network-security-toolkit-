"""The ProgressSink port in the scanner (step 1.1c)."""

import threading

from toolkit.ports import NullProgress, ProgressSink


class RecordingProgress:
    def __init__(self):
        self.events = []
        self.threads = set()
        self._lock = threading.Lock()

    def start(self, total):
        self.events.append(("start", total))

    def advance(self, done=1):
        with self._lock:
            self.events.append(("advance", done))
            self.threads.add(threading.current_thread().name)

    def finish(self):
        self.events.append(("finish",))


def test_sinks_fulfil_the_port(scanner):
    assert isinstance(NullProgress(), ProgressSink)
    assert isinstance(scanner.ConsoleProgress(), ProgressSink)
    assert isinstance(RecordingProgress(), ProgressSink)


def test_scan_reports_every_check(scanner, tcp_server, free_port):
    open_port = tcp_server(b"SSH-2.0-OpenSSH_8.9p1\r\n")
    progress = RecordingProgress()
    results = scanner.scan_ports("127.0.0.1", [open_port, free_port], ["tcp", "udp"], progress)

    assert len(results) == 4
    assert progress.events[0] == ("start", 4) and progress.events[-1] == ("finish",)
    assert progress.events[1:-1] == [("advance", 1)] * 4
    assert threading.main_thread().name not in progress.threads  # called from workers


def test_console_progress_lines(scanner, capsys):
    progress = scanner.ConsoleProgress(interval=60)  # only the first and the last line
    progress.start(4)
    for _ in range(4):
        progress.advance()
    progress.finish()
    out = capsys.readouterr().out
    assert out.startswith("\rProgress: 0.00% complete | 4 tasks remaining")
    assert out.endswith("\rProgress: 100.00% complete | 0 tasks remaining\n"
                        "Progress: 100.00% complete\n")


def test_zero_tasks_no_division_by_zero(scanner, capsys):
    progress = scanner.ConsoleProgress()
    progress.start(0)
    progress.finish()
    assert capsys.readouterr().out == ""
    assert scanner.scan_ports("127.0.0.1", [], ["tcp"], progress) == []
