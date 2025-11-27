import React from 'react';

const ControlToolbar = ({ onStart, onStop, onReset }) => {
  return (
    <div className="absolute bottom-4 right-4 flex space-x-2">
      <button onClick={onStart} className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded">
        Start
      </button>
      <button onClick={onStop} className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded">
        Stop
      </button>
      <button onClick={onReset} className="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded">
        Reset
      </button>
    </div>
  );
};

export default ControlToolbar;
