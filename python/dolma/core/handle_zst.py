from smart_open.compression import tweak_close, register_compressor, get_supported_extensions

def _handle_zst(file_obj, mode):
    import zstandard
    result = zstandard.open(file_obj, mode=mode)
    #  tweak_close(result, file_obj)
    return result


if ".zst" not in get_supported_extensions():
    register_compressor('.zst', _handle_zst)
