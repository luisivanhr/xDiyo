"""Acquire missing badges using the exact provider team IDs in published matches.

Existing assets and provenance are retained. Images are decoded before
acceptance; static GIFs are losslessly encoded as PNG for the report renderer.
No name-based substitution or crest generation is performed.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import urllib.request

from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    catalog = json.loads((HERE / 'catalog.json').read_text(encoding='utf-8'))
    pending = [(key, item) for key, item in catalog.items() if not item.get('badge_path')]

    def fetch(entry):
        key, item = entry
        url = f'https://img.sofascore.com/api/v1/team/{key}/image'
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read(2_000_001)
            if len(payload) > 2_000_000:
                raise ValueError('Badge exceeds renderer embedding limit')
            source_sha256 = hashlib.sha256(payload).hexdigest()
            conversion = {}
            with Image.open(BytesIO(payload)) as image:
                fmt = image.format.lower()
                size = image.size
                image.verify()
            if fmt == 'gif':
                with Image.open(BytesIO(payload)) as image:
                    if getattr(image, 'n_frames', 1) != 1:
                        raise ValueError('Animated badge requires explicit review')
                    buffer = BytesIO()
                    image.convert('RGBA').save(buffer, format='PNG')
                    payload = buffer.getvalue()
                fmt = 'png'
                conversion = {'source_format': 'gif', 'source_sha256': source_sha256,
                              'conversion': 'Single-frame GIF losslessly encoded as RGBA PNG; artwork unchanged.'}
            if fmt not in {'png', 'webp', 'jpeg'} or min(size) < 16:
                raise ValueError(f'Unexpected badge format or dimensions: {fmt} {size}')
            target = HERE / 'badges' / f'{key}.{fmt}'
            target.write_bytes(payload)
            return key, {
                'badge_path': f'badges/{target.name}', 'format': fmt,
                'source_url': url, 'source_collection': 'Sofascore team images',
                'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'identity_match': f'Exact Sofascore team ID {key} from published home_id/away_id',
                'sha256': hashlib.sha256(payload).hexdigest(),
                'width': size[0], 'height': size[1],
                'license': 'No separate image license identified; club artwork and trademark rights remain with their owners.',
                **conversion,
            }, None
        except Exception as exc:
            return key, {}, str(exc)

    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for key, fields, error in pool.map(fetch, pending):
            if error:
                errors.append({'team_id': key, 'name': catalog[key]['name'], 'error': error})
            else:
                catalog[key].update(fields)
    summary = {
        'teams': len(catalog),
        'badges': sum(bool(item.get('badge_path')) for item in catalog.values()),
        'unmatched': [{'team_id': key, 'name': item['name']} for key, item in catalog.items() if not item.get('badge_path')],
        'errors': errors,
    }
    (HERE / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (HERE / 'collection.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
