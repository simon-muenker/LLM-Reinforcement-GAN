import typing

import cltrier_lib


def create_chats(
    batch: typing.List[str], message_role: cltrier_lib.inference.schemas.Roles
) -> typing.List[cltrier_lib.inference.schemas.Chat]:
    return [
        cltrier_lib.inference.schemas.Chat(
            messages=[
                cltrier_lib.inference.schemas.Message(
                    role=message_role,
                    content=sample,
                )
            ]
        )
        for sample in batch
    ]
