import datetime
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

"""UTC to ISO 8601 with Local TimeZone information without microsecond"""


def isotime():
    return (
        datetime.datetime.utcnow()
        .replace(tzinfo=datetime.UTC)
        .astimezone()
        .replace(microsecond=0)
        .isoformat()
    )


class NpEncoder(json.JSONEncoder):
    """
    Custom encoder to serialize Numpy data types.

    [StackOverflow reference](https://stackoverflow.com/q/50916422)
    """

    def default(self, obj):
        """ """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)


def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data.
    Used JSON dumps to form strings, and the blake2b algorithm to hash.

    Parameters
    ----------
    keyed_data : dict
        dict with the keys to generate a fingerprint
    digest_size : int, optional
        Digest size for blake2b hash code, by default 16

    Returns
    -------
    str
        The hexadecimal digest
    """
    h = hashlib.blake2b(digest_size=digest_size)
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()


def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path

    Parameters
    ----------
    path : str
        A path possibly containing environment variables and user (~) shortcut

    Returns
    -------
    str
        The expanded absolute path
    """
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def native_type(value):
    """
    Converts a numpy type to a native python type.
    See:
    https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998
    """
    return getattr(value, "tolist", lambda: value)()


def make_executable(path):
    """
    Makes a file executable.

    https://stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python
    """
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2  # copy R bits to X
    os.chmod(path, mode)


def find_executable(exename=None, envname=None):
    """
    Finds an executable from a given name or environmental variable.

    If neither are files, the path will be searched for exename

    """

    # Simply return if this exists
    if exename and os.path.isfile(exename):
        assert os.access(exename, os.X_OK), f"File is not executable: {exename}"
        return full_path(exename)

    envexe = os.environ.get(envname)
    if envexe and os.path.isfile(envexe):
        assert os.access(envexe, os.X_OK), f"File is not executable: {envexe}"
        return full_path(envexe)

    if not exename and not envname:
        raise ValueError("No exename or envname ")

    # Start searching. Search PATH before the current working directory so a
    # binary planted in the cwd cannot shadow a trusted executable found on PATH
    # (CWE-426, untrusted search path).
    search_path = []
    path_env = os.environ.get("PATH")
    if path_env:
        search_path.append(path_env)
    search_path.append(os.getcwd())
    search_path_str = os.pathsep.join(search_path)
    bin_location = shutil.which(exename, path=search_path_str)

    if bin_location and os.path.isfile(bin_location):
        return full_path(bin_location)

    raise ValueError(f"Could not find executable: exename={exename}, envname={envname}")


def execute(cmd, cwd=None):
    """

    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running

    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")

    Useful in Jupyter notebook

    """
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd
    )
    yield from iter(popen.stdout.readline, "")
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# Alternative execute
def execute2(cmd, timeout=None, cwd=None):
    """
    Execute with time limit (timeout) in seconds, catching run errors.
    """

    output = {"error": True, "log": ""}
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        #  p = subprocess.run(' '.join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        #                     universal_newlines=True, timeout = timeout)
        output["log"] = p.stdout
        output["error"] = False
        output["why_error"] = ""
    except subprocess.TimeoutExpired as ex:
        output["log"] = ex.stdout + "\n" + str(ex)
        output["why_error"] = "timeout"
    except Exception:
        output["log"] = "unknown run error"
        output["why_error"] = "unknown"
    return output
