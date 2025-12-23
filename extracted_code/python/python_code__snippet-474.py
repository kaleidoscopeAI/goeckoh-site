    from pip._vendor.rich import print

    print()

    def test(foo: float, bar: float) -> None:
        list_of_things = [1, 2, 3, None, 4, True, False, "Hello World"]
        dict_of_things = {
            "version": "1.1",
            "method": "confirmFruitPurchase",
            "params": [["apple", "orange", "mangoes", "pomelo"], 1.123],
            "id": "194521489",
        }
        print(render_scope(locals(), title="[i]locals", sort_keys=False))

    test(20.3423, 3.1427)
    print()


