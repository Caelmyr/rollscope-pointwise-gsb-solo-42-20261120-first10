from dataclasses import dataclass

from . import __version__


@dataclass(frozen=True)
class ReportValue:
    path: tuple
    value: int | float

    @property
    def key(self):
        return ".".join(self.path)


def measure_paths(payload, prefix=()):
    for key, value in sorted(payload.items()):
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from measure_paths(value, path)
        elif type(value) in (int, float) or value is None:
            yield ReportValue(path, value)


def summarize_change(previous, current):
    a = {datum.key: datum.value for datum in measure_paths(previous)}
    b = {datum.key: datum.value for datum in measure_paths(current)}
    values = []
    for name in sorted(set(a) | set(b)):
        before = a.get(name)
        after = b.get(name)
        if before == after:
            continue
        change = None
        if before is not None and after is not None:
            change = after - before
        values.append({"measure": name, "before": before, "after": after, "change": change})
    return values


def settings_diff(previous, current):
    old = previous.as_dict()
    fresh = current.as_dict()
    difference = []
    for section in sorted(old):
        for key, value in old[section].items():
            current_value = fresh[section][key]
            if value != current_value:
                difference.append(
                    {
                        "setting": f"{section}.{key}",
                        "before": value,
                        "after": current_value,
                    }
                )
    return difference


def finding_counts(result):
    return {table["key"]: len(table["rows"]) for table in result["tables"]}


def snapshot_comparison(saved, current, settings):
    compatible = saved.version == __version__
    return {
        "snapshot": saved.summary(),
        "version_matched": compatible,
        "changes": summarize_change(saved.result["summary"], current["summary"]),
        "setting_changes": settings_diff(saved.settings, settings),
        "table_sizes": summarize_change(finding_counts(saved.result), finding_counts(current)),
    }
