import json

from squadds.core.json_utils import normalize_setup_payload


def test_normalize_setup_payload_unwraps_single_item_list_wrapped_setup():
    payload = json.dumps([{"setup": {"max_passes": 12}}])

    assert normalize_setup_payload(payload) == {"max_passes": 12}
