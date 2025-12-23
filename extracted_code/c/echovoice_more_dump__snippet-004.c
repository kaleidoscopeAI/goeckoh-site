| |             -^^^^^^ method cannot be called on `FilterMap<IntoIter, fn(...) -> ... {...::ok}>` due to unsatisfied trait bounds
| |_____________|
|
|
= note: the following trait bounds were not satisfied:
        `std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: Stream`
        which is required by `std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: StreamExt`
        `<fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok} as FnOnce<(std::result::Result<walkdir::DirEntry, walkdir::Error>,)>>::Output = std::option::Option<_>`
        which is required by `std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: Iterator`
        `fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}: FnMut<(std::result::Result<walkdir::DirEntry, walkdir::Error>,)>`
        which is required by `std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: Iterator`
        `&std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: Stream`
        which is required by `&std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: StreamExt`
        `&mut std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: Stream`
        which is required by `&mut std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: StreamExt`
        `std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: Iterator`
        which is required by `&mut std::iter::FilterMap<walkdir::IntoIter, fn(std::result::Result<walkdir::DirEntry, universal_engine::error::Error>) -> std::option::Option<walkdir::DirEntry> {std::result::Result::<walkdir::DirEntry, universal_engine::error::Error>::ok}>: Iterator`

