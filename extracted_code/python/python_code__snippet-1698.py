class BrokenRepr:
    def __repr__(self) -> str:
        1 / 0
        return "this will fail"

from typing import NamedTuple

class StockKeepingUnit(NamedTuple):
    name: str
    description: str
    price: float
    category: str
    reviews: List[str]

d = defaultdict(int)
d["foo"] = 5
data = {
    "foo": [
        1,
        "Hello World!",
        100.123,
        323.232,
        432324.0,
        {5, 6, 7, (1, 2, 3, 4), 8},
    ],
    "bar": frozenset({1, 2, 3}),
    "defaultdict": defaultdict(
        list, {"crumble": ["apple", "rhubarb", "butter", "sugar", "flour"]}
    ),
    "counter": Counter(
        [
            "apple",
            "orange",
            "pear",
            "kumquat",
            "kumquat",
            "durian" * 100,
        ]
    ),
    "atomic": (False, True, None),
    "namedtuple": StockKeepingUnit(
        "Sparkling British Spring Water",
        "Carbonated spring water",
        0.9,
        "water",
        ["its amazing!", "its terrible!"],
    ),
    "Broken": BrokenRepr(),
}
data["foo"].append(data)  # type: ignore[attr-defined]

from pip._vendor.rich import print

# print(Pretty(data, indent_guides=True, max_string=20))

class Thing:
    def __repr__(self) -> str:
        return "Hello\x1b[38;5;239m World!"

print(Pretty(Thing()))


