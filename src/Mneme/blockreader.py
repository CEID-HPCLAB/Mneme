import pandas as pd
import math
import os
import pickle
from time import perf_counter as time
from ._mneme_logging import _Mneme_logger

def get_csv_ncols(path: str) -> tuple:
    '''
    Retrieves information about the columns of a CSV file.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - ncols (int): The number of columns in the CSV file.
            - columns (list): A list of column names extracted from the CSV file.
            - features_idxs_map (dict): A dictionary mapping column names to their corresponding indices.

    This function reads the first row of the CSV file located at the specified path to determine the number of columns,
    extract the column names and create a mapping of column names to their indices.

    Example:
        Suppose we have a CSV file with the following contents:
        ```
        Name,Age,Gender
        George,25,Male
        Maria,30,Female
        ```

        Calling `get_csv_ncols('data.csv')` would return:
        (3, ['Name', 'Age', 'Gender'], {'Name': 0, 'Age': 1, 'Gender': 2})
    '''
    
    # Read the first row of the CSV file to get column names
    df = pd.read_csv(path, nrows=0)
    columns = df.columns
    ncols = len(columns)
    
    # Create a mapping of column names to their indices
    col_indices = df.columns.to_series().reset_index(drop = True).to_dict()
    features_idxs_map = {v: k for k, v in col_indices.items()}
    
    return ncols, columns, features_idxs_map


def get_csv_nrows(path: str, chunk_size: int = 4096, method: int = 4) -> int:
    '''
    Retrieves the number of rows in a CSV file.

    Args:
        path (str): The file path to the CSV file.
        chunk_size (int, optional): The size of each chunk used for reading the file in method 2. Default is 4096.
        method (int, optional): The method used for counting rows (Default is 4):
            - 1: Read the entire file into memory and count rows using pandas.
            - 2: Read the file in chunks and sum the row counts of each chunk.
            - 3: Read the file line by line and count rows.
            - 4: Count the number of newline characters in the file using file iteration.
            - 5: Use the 'wc -l' command to count lines in the file. Requires Unix-like system. 

    Returns:
        int: The number of rows in the CSV file.

    This function provides various methods to determine the number of rows in a CSV file, allowing flexibility based on
    file size, system resources, and operating environment. If the specified method is not recognized, it defaults to method 4.

    Example:
        Suppose we have a CSV file named 'data.csv' with the following contents:
        ```
        Name,Age,Gender
        George,25,Male
        Maria,30,Female
        ```

        Calling `get_csv_nrows('data.csv')` would return the total number of rows in the file, excluding the header row.
    '''
    
    # Using pandas to read the entire file and count rows
    if method == 1:
        df = pd.read_csv(path)
        nrows = df.shape[0]
        return nrows

    # Reading the file in chunks and summing the row counts of each chunk
    elif method == 2:
        chunks = pd.read_csv(path, chunksize = chunk_size)
        total_length = 0
        for chunk in chunks:
            total_length += chunk.shape[0]
        return total_length

    # Reading the file line by line and counting rows
    elif method == 3:
        from csv import reader as csv_reader
        opened_file = open(path)
        read_file = csv_reader(opened_file)
        apps_data = list(read_file)
        rowcount = len(apps_data)  
        return rowcount - 1 # Exclude header

    # Counting the number of newline characters in the file using file iteration
    elif method == 4:
        from functools import partial
        f = open(path)
        rowcount = sum(chunk.count('\n') for chunk in iter(partial(f.read, 1 << 15), ''))
        return rowcount - 1  # Exclude header
    
    # Using the 'wc -l' command to count lines in the file
    elif method == 5:
        from subprocess import run
        result = run(['wc', '-l', path], capture_output=True, text=True)
        line_count = int(result.stdout.split()[0])
        return line_count
    
    # Default method if the specified method is not recognized
    else:
        _Mneme_logger.benchmark(f"[INFO [get_csv_nrows()]] Method {method} is not recognized. Defaulting to method 4.")
        return get_csv_nrows(path, method=4)
    

class BlockReader:
    '''
    A class designed to efficiently process large datasets by dividing the file into manageable blocks
    and accurately computing their corresponding offsets.

    This class facilitates the efficient handling of extensive datasets by partitioning the file into
    manageable blocks and precisely calculating their offsets for streamlined processing.
    '''

    def __init__(self, training_file: str, target_label_name: list = [], num_blocks: int = 1, 
                 n_rows: int = -1, read_chunk_size: int = 1000, method: int = 4, 
                 block_offset_cache: str = "", block_offset_save: str = "") -> None:
        '''
        Initialize a BlockReader object.
        
        Args:
            training_file (str): The file path to the training data.
            target_label_name (list, optional): A list containing target label names. Defaults to an empty list.
            num_blocks (int, optional): The desired number of blocks to divide the file into. Defaults to 1.
            n_rows (int, optional): The total number of rows in the file. Defaults to -1, triggering automatic calculation.
            read_chunk_size (int, optional): The size of each reading chunk in bytes. Defaults to 1000.
            method (int, optional): The method utilized to determine the number of rows. Defaults to 4.
            block_offset_cache (str, optional): The path to a cache file containing precomputed block offsets. Defaults to "".
            block_offset_save (str, optional): The path to save precomputed block offsets to a binary file. Defaults to "".

        Raises:
            FileNotFoundError: If the specified training data file is not found.
            ValueError: If the number of blocks provided is negative.
        '''
        
        # Initialize logger 
        self._logger = _Mneme_logger

        # Set properties and parameters
        self.training_file = training_file

        # Check if training file exists
        if not os.path.exists(training_file):
            raise FileNotFoundError(f"File '{training_file}' not found. Please make sure the file exists and try again.")

        # Start timer (computing BatchLoader Initialization time)
        t0 = time()

        # Calculate number of rows if not provided
        if n_rows == -1:
            self._logger.debug(f"Number of rows not provided. Calculating the number of rows...")
            n_rows = get_csv_nrows(training_file, chunk_size = read_chunk_size, method = method)

        # Retrieve number of columns and column information
        n_cols, columns, feature_idxs_map = get_csv_ncols(training_file)

        # Validate num_blocks parameter
        if num_blocks < 0:
            raise ValueError("The number of blocks cannot be negative.")

        # Calculate block size
        block_size = int(math.ceil(n_rows / num_blocks))

        # If no target label names are provided, use the last column of the dataset as the target label
        if len(target_label_name) == 0:
            target_label_name = columns[-1].split()

        # Set attributes
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.columns = columns
        self.feature_idxs_map = feature_idxs_map
        self.target_label_name = [*target_label_name]
        self.block_size = block_size

        # Load cached block offsets if provided
        if len(block_offset_cache) > 0: 
            block_offsets = self._load_cached_offsets(block_offset_cache)
        else:
            # Compute block offsets if not cached
            self._logger.debug(f"Previous block offsets weren't found. Computing block offsets...")
            block_offsets = self.create_block_offset_file(training_file, n_rows, block_size, block_offset_save)

        # Update attributes with block offset information
        num_blocks = len(block_offsets)
        self.num_blocks = num_blocks
        self.block_offsets = block_offsets

        # End timer (computing BatchLoader initialization time)
        t1 = time()

        # Log attribute values (excluding specific attributes)
        not_included_attrs = ['block_offsets', '_logger', 'feature_idxs_map']
        for attr in vars(self):
            if attr not in not_included_attrs:
                self._logger.debug(f"BatchLoader.{attr} = {getattr(self, attr)}")

        # Log initialization time
        self._logger.benchmark(f"BatchLoader Initialization time: {t1 - t0:.6f}")
        

    def create_block_offset_file(self, path: str, nrows: int, block_size: int, block_offset_save: str) -> list:
        '''
        Creates a file with binary offsets based on the provided parameters.

        This function reads through the specified dataset file and calculates the byte offsets
        for each block based on the given block size and total number of rows.

        Args:
            path (str): The path to the training dataset file.
            nrows (int): The total number of rows in the dataset.
            block_size (int): The size of each block (the number of samples (== rows)).
            block_offset_save (str): The path to save the binary offsets file.

        Returns:
            list: A list containing the computed block offsets.

        Raises:
            FileNotFoundError: If the training dataset file does not exist.
        '''
        
        if not (file := open(path, 'r')) :
            raise FileNotFoundError(f"Error when opening file '{path}'. Aborting...")
        
        block_offsets = []
        offset = 0
        for i, line in enumerate(file):
            
            # Store the offset if it marks the start of a new block
            if (i - 1) % block_size == 0 and i != 0:
                block_offsets.append(offset)
            
            # Break loop if the specified number of rows has been reached
            if i == nrows:
                break
                
            offset += len(line)  
        
        # Close the file after reading
        file.close() 
            
        # Save block offsets to file if a save path is provided
        if block_offset_save:
            self._save_offsets(block_offsets, block_offset_save)
        
        # Return the computed block offsets
        return block_offsets 
    

    def _get_offset_name(self, path: str, num_blocks: int) -> str:
        '''
        Returns the filename for storing the computed binary offsets.

        Args:
            path (str): The path of the original file.
            num_blocks (int): The number of blocks used to compute the offsets.

        Returns:
            str: The filename for storing the computed binary offsets.

        This function takes the path of the original file and the number of blocks
        used to compute the offsets and constructs a filename for storing the
        computed binary offsets. The filename is constructed by appending the
        number of blocks to the basename of the original file and adding the ".dat"
        extension.

        Example:
            If the original file path is "data.csv" and num_blocks is 5, the
            returned filename will be "data_5.dat".
        
        [Note]:
            This function is deprecated and not used in the current implementation.
        '''
        
        # Extract the filename from the provided path
        basename = os.path.basename(path)

        # Split the filename and its extension
        filename, _ = os.path.splitext(basename)

        # Construct the offset filename by appending the number of blocks and the .dat extension
        offset_filename = filename + '_{}'.format(num_blocks) + '.dat'

        # Return the constructed offset filename
        return offset_filename


    def _convert_path(self, path: str) -> str:
        '''
        Converts the path format based on the operating system (OS).

        Args:
            path (str): The original path to be converted.

        Returns:
            str: The converted path.

        This function takes a path as input and converts its format based on the
        operating system. For Windows OS (identified by os.name == 'nt'), the function
        replaces forward slashes '/' with backward slashes '\\'. For Linux or Unix-like
        OS, it replaces backward slashes '\\' with forward slashes '/' and removes
        any colons ':' present in the path.

        Example:
            - On a Windows system: 
                If the input path is "C:/Users/Documents/file.txt", 
                the returned path will be "C:\\Users\\Documents\\file.txt".
            - On a Linux or Unix-like system:
                If the input path is "C:\\Users\\Documents\\file.txt",
                the returned path will be "C:/Users/Documents/file.txt".
        '''
    
        if os.name == 'nt':  # Windows OS
            return path.replace('/', '\\')
        
        else:  # Linux or Unix-like OS
            return path.replace('\\', '/').replace(':', '')
        

    def _load_cached_offsets(self, block_offset_file: str) -> list:
        '''
        Loads cached block offsets from a binary file.

        This method attempts to load cached block offsets from a binary file specified
        by the provided block_offset_file path. If the file exists, it reads the
        block offsets using pickle and returns them. If the file does not exist,
        it logs a critical message indicating the absence of the cache file and
        proceeds to create the block offsets using the create_block_offset_file method.

        Args:
            block_offset_file (str): The path to the binary file containing cached block offsets.

        Returns:
            list: The list of cached block offsets.
        '''

        # Convert the provided path to the appropriate format for the current OS
        binary_offset_file = self._convert_path(block_offset_file)
        # Check if the file extension is ".dat", if not, append it
        if not binary_offset_file.endswith('.dat'):
            binary_offset_file += '.dat'

        # Check if the binary offset file exists
        if os.path.exists(binary_offset_file):
            # If the file exists, load the cached block offsets
            with open(binary_offset_file, 'rb') as offset_file:
                block_offsets = pickle.load(offset_file)
            # Log a debug message indicating the use of cached block offsets
            self._logger.debug(f"Using cached block offsets from '{binary_offset_file}'.")
            return block_offsets
        
        else:
            # If the file does not exist, log a critical message and create the block offsets
            self._logger.critical(f"Block offset cache file '{binary_offset_file}' not found. Generating new block offsets.")
            return self.create_block_offset_file(self.training_file, self.n_rows, self.block_size)

    
    def _save_offsets(self, block_offsets: list, block_offset_file: str) -> None:
        '''
        Save block offsets to a binary file (.dat).

        Args:
            block_offsets (list): List of block offsets to be saved.
            block_offset_file (str): Path to the file where block offsets will be saved.

        Returns:
            None

        This method saves the provided list of block offsets to a binary file specified by
        the 'block_offset_file' parameter.

        If the directory where the file should be saved does not exist, an error message is
        logged, indicating that the block offsets were not saved due to the non-existent path.

        If the directory exists, the block offsets are written to the binary file using
        pickle.dump(). A debug message is logged, indicating that the block offsets are being
        written to the file.
        '''
        
        binary_offset_file = self._convert_path(block_offset_file)
        # Add ".dat" extension to the block offset file if it's not already present
        if not binary_offset_file.endswith('.dat'):
            binary_offset_file += '.dat'
        path = os.path.dirname(binary_offset_file)
        
        # Check if the directory exists
        if not os.path.exists(path) :
            self._logger.error(f"Failed to save block offsets. The directory '{path}' does not exist.")
            
        else: 
            # Save block offsets to the binary file
            self._logger.debug(f"Writing block offsets to file: {binary_offset_file}.")
            with open(binary_offset_file, 'wb') as filehandle:
                pickle.dump(block_offsets, filehandle)

    
    def get_num_blocks(self) -> int:
        ''' 
        Returns the number of blocks. 
        '''
        return self.num_blocks  