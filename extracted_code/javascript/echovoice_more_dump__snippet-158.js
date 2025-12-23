168 |         let mut current_energy = graph.total_energy() as f64;
    |                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^ an `as` expression can be used to convert enum types to numeric types only if the enum type is unit-only or field-less
    |
    = note: see https://doc.rust-lang.org/reference/items/enumerations.html#casting for more information

