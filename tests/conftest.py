from _shared.timing import TIMINGS


def pytest_terminal_summary(terminalreporter):  # type: ignore[no-untyped-def]
    if TIMINGS:
        terminalreporter.section("integration timings")
        for line in TIMINGS:
            terminalreporter.write_line(line)
