    We must ensure that the Task::from_pid function is safe and returns a valid task if possible.

    We must ensure that the set_user_nice and other functions are called with valid parameters.

    We must ensure that the try_to_free_mem_cgroup_pages is called with valid parameters and in the right context.

    We note that the kernel module must be built with the appropriate configuration.

