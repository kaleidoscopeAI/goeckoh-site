We note that the Rust for Linux kernel does not have all bindings by default. We may need to generate them or use the existing ones.

We assume that the necessary kernel functions are available via bindings if we have included the headers.

We note that the kernel's task_struct is complex and we are only accessing a few fields.

We must be careful with the unsafe blocks when dereferencing kernel pointers.

We must ensure that the Task::from_pid function is safe and returns a valid task if possible.

We must ensure that the set_user_nice and other functions are called with valid parameters.

We must ensure that the try_to_free_mem_cgroup_pages is called with valid parameters and in the right context.

We note that the kernel module must be built with the appropriate configuration.

