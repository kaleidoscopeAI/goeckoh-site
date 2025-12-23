def __instancecheck__(cls, __instance: Any) -> bool:
    return isinstance(__instance, cls._backported_typevarlike)


