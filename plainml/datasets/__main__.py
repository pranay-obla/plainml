"""Print the example datasets' catalog as JSON, for the website's copy:

python -m plainml.datasets > frontend/lib/samples.json
"""

import json

from plainml.datasets import SAMPLES

print(json.dumps([sample.to_dict() for sample in SAMPLES.values()], indent=2))
