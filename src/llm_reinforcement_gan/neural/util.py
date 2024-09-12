import typing


def remove_prefixes(
    content: typing.List[typing.Any], prefix: typing.List[typing.Any]
) -> typing.List[typing.Any]:
    return [cont[len(pref) :] for pref, cont in zip(prefix, content)]
