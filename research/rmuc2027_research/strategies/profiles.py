"""Optional local strategy display metadata, independent of policy identity."""
import json
from pathlib import Path
def load_profile(source):
    path = Path(source)/'profile.json'
    value = json.loads(path.read_text()) if path.is_file() else {}
    return {'agent': value.get('agent', {'provider':'human','model':'local'}), **({'display_name':value['display_name']} if value.get('display_name') else {})}
