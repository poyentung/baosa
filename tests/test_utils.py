import os
import numpy as np
import pytest
from balsa.utils import Tracker

@pytest.fixture
def temp_folder(tmp_path):
    """Create a temporary folder for testing."""
    return str(tmp_path / "test_tracker")

def test_tracker_initialization(temp_folder):
    """Test the initialization of the Tracker class."""
    tracker = Tracker(temp_folder)
    assert tracker.foldername == temp_folder
    assert tracker.counter == 0
    assert tracker.results == []
    assert tracker.x_values == []
    assert tracker.current_best == float("inf")
    assert tracker.current_best_x is None
    assert os.path.exists(temp_folder)

def test_tracker_dump_trace(temp_folder):
    """Test the dump_trace method."""
    tracker = Tracker(temp_folder)
    tracker.results = [1.0, 2.0, 3.0]
    tracker.dump_trace()
    
    result_file = os.path.join(temp_folder, "result.npy")
    assert os.path.exists(result_file)
    loaded_results = np.load(result_file)
    np.testing.assert_array_equal(loaded_results, np.array([1.0, 2.0, 3.0]))

def test_tracker_track(temp_folder, capsys):
    """Test the track method."""
    tracker = Tracker(temp_folder)
    
    # Test tracking with improvement
    tracker.track(2.0, np.array([1.0, 2.0]))
    assert tracker.counter == 1
    assert tracker.current_best == 2.0
    np.testing.assert_array_equal(tracker.current_best_x, np.array([1.0, 2.0]))
    assert tracker.results == [2.0]
    assert len(tracker.x_values) == 1
    np.testing.assert_array_equal(tracker.x_values[0], np.array([1.0, 2.0]))
    
    # Test tracking without improvement
    tracker.track(3.0, np.array([3.0, 4.0]))
    assert tracker.counter == 2
    assert tracker.current_best == 2.0
    np.testing.assert_array_equal(tracker.current_best_x, np.array([1.0, 2.0]))
    assert tracker.results == [2.0, 2.0]
    assert len(tracker.x_values) == 2
    
    # Check if output is correct
    captured = capsys.readouterr()
    assert "current best f(x): 2.0" in captured.out
    assert "current best x: [1. 2.]" in captured.out

def test_tracker_track_with_saver(temp_folder):
    """Test the track method with saver option."""
    tracker = Tracker(temp_folder)
    tracker.track(1.0, np.array([1.0]), saver=True)
    
    result_file = os.path.join(temp_folder, "result.npy")
    assert os.path.exists(result_file)

def test_tracker_track_auto_save(temp_folder):
    """Test the track method's auto-save functionality."""
    tracker = Tracker(temp_folder)
    for i in range(21):
        tracker.track(float(i))
    
    result_file = os.path.join(temp_folder, "result.npy")
    assert os.path.exists(result_file)

def test_tracker_track_zero_result(temp_folder):
    """Test the track method when result is zero."""
    tracker = Tracker(temp_folder)
    tracker.track(0.0, np.array([0.0]))
    
    result_file = os.path.join(temp_folder, "result.npy")
    assert os.path.exists(result_file)
