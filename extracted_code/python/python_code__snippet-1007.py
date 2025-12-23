    from dataclasses import dataclass

    @dataclass
    class E:

        size: Optional[int] = None
        ratio: int = 1
        minimum_size: int = 1

    resolved = ratio_resolve(110, [E(None, 1, 1), E(None, 1, 1), E(None, 1, 1)])
    print(sum(resolved))


