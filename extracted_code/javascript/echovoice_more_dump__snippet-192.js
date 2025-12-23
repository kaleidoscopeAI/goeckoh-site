let result = unsafe { set_user_nice(task_ptr, new_nice.clamp(-20, 19)) };
