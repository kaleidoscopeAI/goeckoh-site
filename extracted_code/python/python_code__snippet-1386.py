import platform
ver = platform.libc_ver()
result = []
if ver[0] == 'glibc':
    for s in ver[1].split('.'):
        result.append(int(s) if s.isdigit() else 0)
    result = tuple(result)
return result


