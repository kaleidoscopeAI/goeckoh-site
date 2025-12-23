"""
Print the information from installed distributions found.
"""
results_printed = False
for i, dist in enumerate(distributions):
    results_printed = True
    if i > 0:
        write_output("---")

    write_output("Name: %s", dist.name)
    write_output("Version: %s", dist.version)
    write_output("Summary: %s", dist.summary)
    write_output("Home-page: %s", dist.homepage)
    write_output("Author: %s", dist.author)
    write_output("Author-email: %s", dist.author_email)
    write_output("License: %s", dist.license)
    write_output("Location: %s", dist.location)
    if dist.editable_project_location is not None:
        write_output(
            "Editable project location: %s", dist.editable_project_location
        )
    write_output("Requires: %s", ", ".join(dist.requires))
    write_output("Required-by: %s", ", ".join(dist.required_by))

    if verbose:
        write_output("Metadata-Version: %s", dist.metadata_version)
        write_output("Installer: %s", dist.installer)
        write_output("Classifiers:")
        for classifier in dist.classifiers:
            write_output("  %s", classifier)
        write_output("Entry-points:")
        for entry in dist.entry_points:
            write_output("  %s", entry.strip())
        write_output("Project-URLs:")
        for project_url in dist.project_urls:
            write_output("  %s", project_url)
    if list_files:
        write_output("Files:")
        if dist.files is None:
            write_output("Cannot locate RECORD or installed-files.txt")
        else:
            for line in dist.files:
                write_output("  %s", line.strip())
return results_printed


