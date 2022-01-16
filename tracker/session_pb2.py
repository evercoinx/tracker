# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: session.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="session.proto",
    package="evercoinx.session",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\rsession.proto\x12\x11\x65vercoinx.session"\xde\x01\n\x0c\x46rameRequest\x12\x14\n\x0cwindow_index\x18\x01 \x01(\x05\x12\x13\n\x0b\x66rame_index\x18\x02 \x01(\x05\x12\x13\n\x0bhand_number\x18\x03 \x01(\x05\x12\x11\n\thand_time\x18\x04 \x01(\t\x12+\n\ttotal_pot\x18\x05 \x01(\x0b\x32\x18.evercoinx.session.Money\x12\x17\n\x0f\x64\x65\x61ler_position\x18\x06 \x01(\x05\x12&\n\x05seats\x18\x07 \x03(\x0b\x32\x17.evercoinx.session.Seat\x12\r\n\x05\x62oard\x18\x08 \x03(\t"\x0e\n\x0c\x45mptyReponse"y\n\x05Money\x12\x33\n\x08\x63urrency\x18\x01 \x01(\x0e\x32!.evercoinx.session.Money.Currency\x12\x0e\n\x06\x61mount\x18\x02 \x01(\x01"+\n\x08\x43urrency\x12\t\n\x05UNSET\x10\x00\x12\x08\n\x04\x45URO\x10\x01\x12\n\n\x06\x44OLLAR\x10\x02"\x8b\x01\n\x04Seat\x12\x0e\n\x06number\x18\x01 \x01(\x11\x12\x0e\n\x06\x61\x63tion\x18\x02 \x01(\t\x12\'\n\x05stake\x18\x03 \x01(\x0b\x32\x18.evercoinx.session.Money\x12)\n\x07\x62\x61lance\x18\x04 \x01(\x0b\x32\x18.evercoinx.session.Money\x12\x0f\n\x07playing\x18\x05 \x01(\x08\x32Z\n\x07Session\x12O\n\tSendFrame\x12\x1f.evercoinx.session.FrameRequest\x1a\x1f.evercoinx.session.EmptyReponse"\x00\x62\x06proto3',
)


_MONEY_CURRENCY = _descriptor.EnumDescriptor(
    name="Currency",
    full_name="evercoinx.session.Money.Currency",
    filename=None,
    file=DESCRIPTOR,
    create_key=_descriptor._internal_create_key,
    values=[
        _descriptor.EnumValueDescriptor(
            name="UNSET",
            index=0,
            number=0,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="EURO",
            index=1,
            number=1,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.EnumValueDescriptor(
            name="DOLLAR",
            index=2,
            number=2,
            serialized_options=None,
            type=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=355,
    serialized_end=398,
)
_sym_db.RegisterEnumDescriptor(_MONEY_CURRENCY)


_FRAMEREQUEST = _descriptor.Descriptor(
    name="FrameRequest",
    full_name="evercoinx.session.FrameRequest",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="window_index",
            full_name="evercoinx.session.FrameRequest.window_index",
            index=0,
            number=1,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="frame_index",
            full_name="evercoinx.session.FrameRequest.frame_index",
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="hand_number",
            full_name="evercoinx.session.FrameRequest.hand_number",
            index=2,
            number=3,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="hand_time",
            full_name="evercoinx.session.FrameRequest.hand_time",
            index=3,
            number=4,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="total_pot",
            full_name="evercoinx.session.FrameRequest.total_pot",
            index=4,
            number=5,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="dealer_position",
            full_name="evercoinx.session.FrameRequest.dealer_position",
            index=5,
            number=6,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="seats",
            full_name="evercoinx.session.FrameRequest.seats",
            index=6,
            number=7,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="board",
            full_name="evercoinx.session.FrameRequest.board",
            index=7,
            number=8,
            type=9,
            cpp_type=9,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=37,
    serialized_end=259,
)


_EMPTYREPONSE = _descriptor.Descriptor(
    name="EmptyReponse",
    full_name="evercoinx.session.EmptyReponse",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=261,
    serialized_end=275,
)


_MONEY = _descriptor.Descriptor(
    name="Money",
    full_name="evercoinx.session.Money",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="currency",
            full_name="evercoinx.session.Money.currency",
            index=0,
            number=1,
            type=14,
            cpp_type=8,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="amount",
            full_name="evercoinx.session.Money.amount",
            index=1,
            number=2,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[
        _MONEY_CURRENCY,
    ],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=277,
    serialized_end=398,
)


_SEAT = _descriptor.Descriptor(
    name="Seat",
    full_name="evercoinx.session.Seat",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="number",
            full_name="evercoinx.session.Seat.number",
            index=0,
            number=1,
            type=17,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="action",
            full_name="evercoinx.session.Seat.action",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="stake",
            full_name="evercoinx.session.Seat.stake",
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="balance",
            full_name="evercoinx.session.Seat.balance",
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="playing",
            full_name="evercoinx.session.Seat.playing",
            index=4,
            number=5,
            type=8,
            cpp_type=7,
            label=1,
            has_default_value=False,
            default_value=False,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=401,
    serialized_end=540,
)

_FRAMEREQUEST.fields_by_name["total_pot"].message_type = _MONEY
_FRAMEREQUEST.fields_by_name["seats"].message_type = _SEAT
_MONEY.fields_by_name["currency"].enum_type = _MONEY_CURRENCY
_MONEY_CURRENCY.containing_type = _MONEY
_SEAT.fields_by_name["stake"].message_type = _MONEY
_SEAT.fields_by_name["balance"].message_type = _MONEY
DESCRIPTOR.message_types_by_name["FrameRequest"] = _FRAMEREQUEST
DESCRIPTOR.message_types_by_name["EmptyReponse"] = _EMPTYREPONSE
DESCRIPTOR.message_types_by_name["Money"] = _MONEY
DESCRIPTOR.message_types_by_name["Seat"] = _SEAT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FrameRequest = _reflection.GeneratedProtocolMessageType(
    "FrameRequest",
    (_message.Message,),
    {
        "DESCRIPTOR": _FRAMEREQUEST,
        "__module__": "session_pb2"
        # @@protoc_insertion_point(class_scope:evercoinx.session.FrameRequest)
    },
)
_sym_db.RegisterMessage(FrameRequest)

EmptyReponse = _reflection.GeneratedProtocolMessageType(
    "EmptyReponse",
    (_message.Message,),
    {
        "DESCRIPTOR": _EMPTYREPONSE,
        "__module__": "session_pb2"
        # @@protoc_insertion_point(class_scope:evercoinx.session.EmptyReponse)
    },
)
_sym_db.RegisterMessage(EmptyReponse)

Money = _reflection.GeneratedProtocolMessageType(
    "Money",
    (_message.Message,),
    {
        "DESCRIPTOR": _MONEY,
        "__module__": "session_pb2"
        # @@protoc_insertion_point(class_scope:evercoinx.session.Money)
    },
)
_sym_db.RegisterMessage(Money)

Seat = _reflection.GeneratedProtocolMessageType(
    "Seat",
    (_message.Message,),
    {
        "DESCRIPTOR": _SEAT,
        "__module__": "session_pb2"
        # @@protoc_insertion_point(class_scope:evercoinx.session.Seat)
    },
)
_sym_db.RegisterMessage(Seat)


_SESSION = _descriptor.ServiceDescriptor(
    name="Session",
    full_name="evercoinx.session.Session",
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=542,
    serialized_end=632,
    methods=[
        _descriptor.MethodDescriptor(
            name="SendFrame",
            full_name="evercoinx.session.Session.SendFrame",
            index=0,
            containing_service=None,
            input_type=_FRAMEREQUEST,
            output_type=_EMPTYREPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_SESSION)

DESCRIPTOR.services_by_name["Session"] = _SESSION

# @@protoc_insertion_point(module_scope)
