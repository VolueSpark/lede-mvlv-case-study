from typing import List
import hashlib, uuid


def create_hash_from_hex(uuid_list: List[str]):
    # Concatenate the two hex values
    combined_hex = hex(sum([ int(u.replace('-',''), 16) for u in uuid_list]))[2:]

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Update the hash object with the bytes of the combined hex
    hash_object.update(bytes.fromhex(combined_hex))

    # Get the resulting hash as a hex string
    return str(uuid.UUID(hash_object.hexdigest()[-32:]))