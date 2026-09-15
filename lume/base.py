import copy
import os
import tempfile
import warnings
from abc import ABC, abstractmethod

import yaml

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

from lume.serializers.hdf5 import HDF5Serializer

from . import tools


class Base(ABC):
    """
    Base Interface for LUME-compatible code.

    Parameters
    ----------
    input_file : str, optional
        The input file to be used, by default None
    initial_particles : dict, optional
        Initial Particle metadata to be used, by default None
    verbose : bool, optional
        Whether or not to produce verbose output, by default False
    timeout : float, optional
        The timeout in seconds to be used, by default None
    """

    def __init__(
        self,
        input_file=None,
        *,
        initial_particles=None,
        verbose=False,
        timeout=None,
        **kwargs,
    ):

        self._input_file = input_file
        self._initial_particles = initial_particles
        self._input = None
        self._output = None

        # Execution
        self._timeout = timeout

        # Logging
        self._verbose = verbose

        # State
        self._configured = False
        self._finished = False
        self._error = False

    @property
    def input(self):
        """
        Input data as a dictionary
        """
        return self._input

    @input.setter
    def input(self, input):
        self._input = input

    @property
    def output(self):
        """
        require openPMD standard, in the future we can add more methods
        for libs such as pandas Dataframes, xarray DataArrays and Dask Arrays.
        """
        return self._output

    @output.setter
    def output(self, output):
        self._output = output

    @property
    def initial_particles(self):
        """
        Initial Particles
        """
        return self._initial_particles

    @initial_particles.setter
    def initial_particles(self, initial_particles):
        self._initial_particles = initial_particles

    @abstractmethod
    def configure(self):
        """
        Configure and set up for run.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """
        Execute the code.
        """
        raise NotImplementedError

    @property
    def verbose(self):
        """
        Read or configure the verbose flag.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @property
    def timeout(self):
        """
        Read or configure the timeout in seconds.
        """
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        self._timeout = timeout

    @property
    def configured(self):
        """
        Get or set the configured flag state.
        """
        return self._configured

    @configured.setter
    def configured(self, configured):
        self._configured = configured

    @property
    def finished(self):
        """
        Get or set the finished flag state.
        """
        return self._finished

    @finished.setter
    def finished(self, finished):
        self._finished = finished

    @property
    def error(self):
        """
        Get or set the error flag state.
        """
        return self._error

    @error.setter
    def error(self, error):
        self._error = error

    @property
    def input_file(self):
        """
        Get or set the input file to be processed.
        """
        return self._input_file

    @input_file.setter
    def input_file(self, input_file):
        """dictionary with parameters?"""
        self._input_file = input_file

    def fingerprint(self):
        """
        Data fingerprint (hash) using the input parameters.

        Returns
        -------
        fingerprint : str
            The hash for this object based on the input parameters.
        """
        return tools.fingerprint(self.input)

    def copy(self):
        """
        Returns a deep copy of this object.

        If a tempdir is being used, will clear this and deconfigure.
        """
        other = copy.deepcopy(self)
        other.reset()
        return other

    def reset(self):
        """
        Reset this object to its initial state.
        """
        pass

    def vprint(self, *args, **kwargs):
        # Verbose print
        if self._verbose:
            print(*args, **kwargs)

    @classmethod
    def from_yaml(cls, yaml_file, parse_input=False):
        """
        Returns an object instantiated from a YAML config file

        Will load intial_particles from an h5 file.

        """
        # Try file
        if os.path.exists(tools.full_path(yaml_file)):
            yaml_file = tools.full_path(yaml_file)
            config = yaml.safe_load(open(yaml_file))
            # Relative paths in the config are resolved against the YAML file's
            # directory.
            config_dir, _ = os.path.split(yaml_file)

            if "input_file" in config:
                # Check that the input file is absolute path...
                # require absolute/ relative to working dir for model input file
                f = os.path.expandvars(config["input_file"])
                if not os.path.isabs(f):
                    config["input_file"] = os.path.join(config_dir, f)

                # Here, we update the config with the input_file contents
                # provided that the input_parser method has been implemented on the subclass
                if parse_input:
                    parsed_input = cls.input_parser(config["input_file"])
                    config.update(parsed_input)

        else:
            # Try raw string. There is no source file, so relative paths are
            # resolved against the current working directory.
            config = yaml.safe_load(yaml_file)
            config_dir = os.getcwd()
            if parse_input and "input_file" in config:
                parsed_input = cls.input_parser(config["input_file"])
                config.update(parsed_input)

        # Form ParticleGroup from file
        if "initial_particles" in config:
            f = config["initial_particles"]
            if not os.path.isabs(f):
                f = os.path.join(config_dir, f)
            config["initial_particles"] = ParticleGroup(f)

        return cls(**config)

    def to_hdf5(self, filename: str) -> None:
        """Serialize an object to an hdf5 file.

        Parameters
        ----------
        filename: str

        """
        serializer = HDF5Serializer()
        serializer.serialize(filename, self)

    @classmethod
    def from_hdf5(cls, filename: str) -> "Base":
        """Load an object from and hdf5.

        Parameters
        ----------
        filename: str

        """
        serializer = HDF5Serializer()
        return serializer.deserialize(filename)

    @abstractmethod
    def archive(self, h5=None):
        """
        Dump inputs and outputs into HDF5 file.

        Parameters
        ----------
        h5 : str or h5py.File
            The filename or handle to HDF5 file in which to write the information.
            If not in informed, a new file is generated.

        Returns
        -------
        h5 : h5py.File
            Handle to the HDF5 file.
        """
        raise NotImplementedError

    @abstractmethod
    def load_archive(self, h5, configure=True):
        """
        Loads input and output from archived h5 file.

        Parameters
        ----------
        h5 : str or h5py.File
            The filename or handle on h5py.File from which to load input and output data
        configure : bool, optional
            Whether or not to invoke the configure method after loading, by default True
        """
        raise NotImplementedError


class CommandWrapper(Base):
    """
    Interface for LUME-compatible code.

    Parameters
    ----------
    input_file : str, optional
        The input file to be used, by default None
    initial_particles : dict, optional
        Initial Particle metadata to be used, by default None
    command : str, optional
        The command to be executed by this wrapper. E.g. ImpactTexe
        If not specified, the class attribute `COMMAND` is used, by default None
    command_mpi : str, optional
        The command to be executed by this wrapper when using MPI. E.g. ImpactTexe-mpi
        If not specified, the class attribute `COMMAND_MPI` is used, by default None
    use_mpi : bool, optional
        Whether or not to use MPI when running this code, by default False
    mpi_run : str, optional
        The command syntax to invoke mpirun. If not specified, the class attribute `MPI_RUN` is used.
        This is expected to be a formated string taking as parameters the number of processors (nproc) and
        the command to be executed (command_mpi), by default None
    use_temp_dir : bool, optional
        Whether or not to use a temporary directory to run the process, by default True
    workdir : str, optional
        The work directory to be used, by default None
    verbose : bool, optional
        Whether or not to produce verbose output, by default False
    timeout : float, optional
        The timeout in seconds to be used, by default None
    """

    COMMAND = ""
    COMMAND_MPI = ""
    MPI_RUN = "mpirun -n {nproc} {command_mpi}"
    WORKDIR = None

    def __init__(
        self,
        input_file=None,
        *,
        initial_particles=None,
        command=None,
        command_mpi=None,
        use_mpi=False,
        mpi_run="",
        use_temp_dir=True,
        workdir=None,
        verbose=False,
        timeout=None,
    ):
        super().__init__(
            input_file=input_file,
            initial_particles=initial_particles,
            verbose=verbose,
            timeout=timeout,
        )
        # Execution
        self._command = command or self.COMMAND
        self._command_mpi = command_mpi or self.COMMAND_MPI
        self._use_mpi = use_mpi
        self._mpi_run = mpi_run or self.MPI_RUN

        self._tempdir = None
        self._use_temp_dir = use_temp_dir
        self._workdir = workdir or self.WORKDIR

        self._base_path = None

    @property
    def use_mpi(self):
        """
        Whether or not MPI should be used if supported.
        """
        return self._use_mpi

    @use_mpi.setter
    def use_mpi(self, use_mpi):
        self._use_mpi = use_mpi

    @property
    def mpi_run(self):
        """
        The command syntax to invoke mpirun. If not specified, the class attribute `MPI_RUN` is used.
        This is expected to be a formated string taking as parameters the number of processors (nproc) and
        the command to be executed (command_mpi).
        """
        return self._mpi_run

    @mpi_run.setter
    def mpi_run(self, mpi_run):
        self._mpi_run = mpi_run

    @property
    def path(self):
        """
        The base path used by the code to manipulate files.
        """
        return self._base_path

    @path.setter
    def path(self, path):
        self._base_path = path

    @property
    def use_temp_dir(self):
        """
        Whether or not the code is using temporary dir to run.

        Returns
        -------
        bool
        """
        return self._use_temp_dir

    @property
    def command(self):
        """
        Get or set the command to be executed. Defaults to `COMMAND`.
        """
        return self._command

    @command.setter
    def command(self, command):
        cmd = command
        if command:
            cmd = tools.full_path(command)
            assert os.path.exists(cmd), "ERROR: Command does not exist:" + command
        self._command = cmd

    @property
    def command_mpi(self):
        """
        Get or set the command to be executed when running with MPI. Defaults to `COMMAND_MPI`.
        """
        return self._command_mpi

    @command_mpi.setter
    def command_mpi(self, command_mpi):
        cmd = command_mpi
        if command_mpi:
            cmd = tools.full_path(command_mpi)
            assert os.path.exists(cmd), "ERROR: Command does not exist:" + command_mpi
        self._command_mpi = cmd

    def get_run_script(self, write_to_path=True):
        """
        Assembles the run script. Optionally writes a file 'run' with this line to path.

        This expect to run with .path as the cwd.

        Parameters
        ----------
        write_to_path : bool
            Whether or not to write the script to the path.

        Returns
        -------
        runscript : str
            The script to run the command.
        """
        _, infile = os.path.split(
            self.input_file
        )  # Expect to run locally. Astra has problems with long paths.

        runscript = [self.command, infile]

        if write_to_path:
            with open(os.path.join(self.path, "run"), "w") as f:
                f.write(" ".join(runscript))

        return runscript

    @classmethod
    def from_archive(cls, archive_h5):
        """
        Class method to return a new instance via restore of an archive file.

        Parameters
        ----------
        archive_h5 : str or h5py.File
            The filename or handle to HDF5 file in which to write the information.

        Returns
        -------
        c : object
            An instance of the class with information from the archive file.
        """
        c = cls()
        c.load_archive(archive_h5)
        return c

    @abstractmethod
    def plot(
        self,
        y=[],
        x=None,
        xlim=None,
        ylim=None,
        ylim2=None,
        y2=[],
        nice=True,
        include_layout=True,
        include_labels=False,
        include_particles=True,
        include_legend=True,
        return_figure=False,
    ):
        """
        Plots output multiple keys.

        Parameters
        ----------
        y : list
            List of keys to be displayed on the Y axis
        x : str
            Key to be displayed as X axis
        xlim : list
            Limits for the X axis
        ylim : list
            Limits for the Y axis
        ylim2 : list
            Limits for the secondary Y axis
        y2 : list
            List of keys to be displayed on the secondary Y axis
        nice : bool
            Whether or not a nice SI prefix and scaling will be used to
            make the numbers reasonably sized. Default: True
        include_layout : bool
            Whether or not to include a layout plot at the bottom. Default: True
        include_labels : bool
            Whether or not the layout will include element labels. Default: False
        include_particles : bool
            Whether or not to plot the particle statistics as dots. Default: True
        include_legend : bool
            Whether or not the plot should include the legend. Default: True
        return_figure : bool
            Whether or not to return the figure object for further manipulation.
            Default: True
        kwargs : dict
            Extra arguments can be passed to the specific plotting function.

        Returns
        -------
        fig : matplotlib.pyplot.figure.Figure
            The plot figure for further customizations or `None` if `return_figure` is set to False.
        """
        raise NotImplementedError

    @abstractmethod
    def write_input(self, input_filename):
        """
        Write the input parameters into the file.

        Parameters
        ----------
        input_filename : str
            The file in which to write the input parameters
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def input_parser(path):
        """
        Invoke the specialized input parser and returns the
        input dictionary.

        Parameters
        ----------
        path : str
            Path to the input file

        Returns
        -------
        input : dict
            The input dictionary
        """
        raise NotImplementedError

    def load_input(self, input_filepath, **kwargs):
        """
        Invoke the `input_parser` with the given input file path as argument.
        This method sets the input property to the contents of the input file after the parser.

        Parameters
        ----------
        input_filepath : str
            The input file path
        kwargs : dict
            Support for extra arguments.
        """
        f = tools.full_path(input_filepath)
        self.original_path, self.original_input_file = os.path.split(
            f
        )  # Get original path, filename
        self.input = self.input_parser(f)

    @abstractmethod
    def load_output(self, **kwargs):
        """
        Reads and load into `.output` the outputs generated by the code.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset this object to its initial state.
        """
        super().reset()
        # Clear this
        if self._use_temp_dir:
            self._base_path = None
            self._configured = False

    @property
    def workdir(self):
        """
        Get or set the working directory
        """
        return self._workdir

    @workdir.setter
    def workdir(self, workdir):
        workdir = tools.full_path(workdir)
        self.setup_workdir(workdir)

    def setup_workdir(self, workdir, cleanup=True):
        """
        Set up the work directory if `use_temp_dir` is set.

        workdir and use_temp_dir: Set up temorary directory inside workdir

        Parameters
        ----------
        workdir : str
            The directory name.
        cleanup : bool
            Whether or not to remove the directory at exit. Defaults to True.
        """

        if not cleanup:
            warnings.warn("cleanup option has been removed", DeprecationWarning)

        # Set paths
        if self._use_temp_dir:
            # Need to attach this to the object. Otherwise it will go out of scope.
            self._tempdir = tempfile.TemporaryDirectory(dir=workdir)
            self._base_path = self._tempdir.name
        elif workdir:
            self._base_path = workdir
        else:
            # Work in place
            self._base_path = self.original_path
