"""Records failing tests (call/setup/teardown failures, subtest failures, collection errors)."""
import json, os
FAILED, COLLECT = set(), set()


def pytest_runtest_logreport(report):
    if report.failed:
        FAILED.add(report.nodeid.split("[")[0] if "::" in report.nodeid else report.nodeid)


def pytest_collectreport(report):
    if report.failed:
        COLLECT.add(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    with open(os.environ["FAIL_OUT"], "w") as f:
        json.dump({"failed": sorted(FAILED), "collect": sorted(COLLECT), "exit": int(exitstatus)}, f)
