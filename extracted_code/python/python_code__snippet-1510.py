# hard-float ABI can be detected from the ELF header of the running
# process
# https://static.docs.arm.com/ihi0044/g/aaelf32.pdf
elf_header = _get_elf_header()
if elf_header is None:
    return False
result = elf_header.e_ident_class == elf_header.ELFCLASS32
result &= elf_header.e_ident_data == elf_header.ELFDATA2LSB
result &= elf_header.e_machine == elf_header.EM_ARM
result &= (
    elf_header.e_flags & elf_header.EF_ARM_ABIMASK
) == elf_header.EF_ARM_ABI_VER5
result &= (
    elf_header.e_flags & elf_header.EF_ARM_ABI_FLOAT_HARD
) == elf_header.EF_ARM_ABI_FLOAT_HARD
return result


