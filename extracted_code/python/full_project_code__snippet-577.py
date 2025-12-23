"""
Given a bytestring, create a CFData object from it. This CFData object must
be CFReleased by the caller.
"""
return CoreFoundation.CFDataCreate(
    CoreFoundation.kCFAllocatorDefault, bytestring, len(bytestring)
)


