import json
from pathlib import Path
from .models import Dataset
from .validation import validate_dataset

TABLES = ('entities', 'relationships', 'events', 'event_links', 'observations', 'ground_truth', 'schedule_revisions', 'delivery_plans', 'blocking_conditions', 'risk_assessments', 'estimates', 'execution_budgets', 'orders', 'commercial_changes', 'acceptances', 'invoices', 'invoice_assessments', 'payments', 'commercial_observations')


def write_dataset(data, directory):
    validate_dataset(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        (directory / f'{name}.jsonl').write_text(''.join(row.model_dump_json() + '\n' for row in getattr(data, name)), encoding='utf-8')
    (directory / 'metadata.json').write_text(data.metadata.model_dump_json(indent=2) + '\n', encoding='utf-8')


def read_dataset(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or metadata.get('schema_version') != '0.7':
        raise ValueError('v0.7 validation requires schema_version 0.7; use migrate-v06 for validated v0.6 datasets')
    raw = {'metadata': metadata}
    for name in TABLES:
        rows = []
        for number, line in enumerate((directory / f'{name}.jsonl').read_text(encoding='utf-8').splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f'{name}.jsonl:{number}: {exc}') from exc
        raw[name] = rows
    data = Dataset.model_validate(raw)
    validate_dataset(data)
    return data
