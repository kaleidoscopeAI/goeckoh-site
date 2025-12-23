if link.is_vcs:
    vcs_backend = vcs.get_backend_for_scheme(link.scheme)
    assert vcs_backend
    url, requested_revision, _ = vcs_backend.get_url_rev_and_auth(
        link.url_without_fragment
    )
    # For VCS links, we need to find out and add commit_id.
    if link_is_in_wheel_cache:
        # If the requested VCS link corresponds to a cached
        # wheel, it means the requested revision was an
        # immutable commit hash, otherwise it would not have
        # been cached. In that case we don't have a source_dir
        # with the VCS checkout.
        assert requested_revision
        commit_id = requested_revision
    else:
        # If the wheel was not in cache, it means we have
        # had to checkout from VCS to build and we have a source_dir
        # which we can inspect to find out the commit id.
        assert source_dir
        commit_id = vcs_backend.get_revision(source_dir)
    return DirectUrl(
        url=url,
        info=VcsInfo(
            vcs=vcs_backend.name,
            commit_id=commit_id,
            requested_revision=requested_revision,
        ),
        subdirectory=link.subdirectory_fragment,
    )
elif link.is_existing_dir():
    return DirectUrl(
        url=link.url_without_fragment,
        info=DirInfo(),
        subdirectory=link.subdirectory_fragment,
    )
else:
    hash = None
    hash_name = link.hash_name
    if hash_name:
        hash = f"{hash_name}={link.hash}"
    return DirectUrl(
        url=link.url_without_fragment,
        info=ArchiveInfo(hash=hash),
        subdirectory=link.subdirectory_fragment,
    )


