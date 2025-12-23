def _self_version_check_logic(
    *,
    state: SelfCheckState,
    current_time: datetime.datetime,
    local_version: DistributionVersion,
    get_remote_version: Callable[[], Optional[str]],
