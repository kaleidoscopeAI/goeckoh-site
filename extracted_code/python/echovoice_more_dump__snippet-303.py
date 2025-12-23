284 |       fn par_iter(&'data self) -> Self::Iter;
    |          -------- the method is available for `Vec<&str>` here
    |
    = help: items from traits can only be used if the trait is in scope
help: trait `IntoParallelRefIterator` which provides `par_iter` is implemented but not in scope; perhaps you want to import it
    |
