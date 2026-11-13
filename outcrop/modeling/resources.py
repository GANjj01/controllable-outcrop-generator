import hashlib
import os
import stat
from pathlib import Path
from .schema import VerifiedResource,ResourceRef,ResourceSet
from .errors import ModelError,invalid


def reject_links(path):
    path=Path(os.path.abspath(path))
    for p in (path,*path.parents):
        try:s=os.lstat(p)
        except (FileNotFoundError,NotADirectoryError):continue
        if stat.S_ISLNK(s.st_mode) or getattr(s,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",1024):
            invalid("resource.path",f"link/reparse point forbidden: {p}")
    return path


def load_resource(ref,root):
    path=confined_path(root,ref.path)
    try:payload=path.read_bytes()
    except OSError as e:raise ModelError(ref.id,"resource_mismatch",str(e)) from e
    if hashlib.sha256(payload).hexdigest()!=ref.sha256:raise ModelError(ref.id,"resource_mismatch","sha256 mismatch")
    return VerifiedResource(ref,payload)


def confined_path(root,name):
    root=reject_links(root)
    if not isinstance(name,str) or not name:invalid('path','nonempty confined relative path required')
    relative=Path(name)
    if relative.anchor or '..' in relative.parts or any(':' in p for p in relative.parts):invalid('path','Windows anchors, traversal and alternate streams forbidden')
    target=reject_links(root/relative)
    if not target.is_relative_to(root):invalid('path','path escaped declared root')
    return target


def load_resources(spec,root):
    from .schema import thaw
    return ResourceSet({r["id"]:load_resource(ResourceRef(**r),root) for r in thaw(spec.data).get("resources",[])})
