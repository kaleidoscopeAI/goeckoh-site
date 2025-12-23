import React, { useState } from 'react';

interface ExportData {
  sessions: any[];
  settings: any;
  progress: any;
  timestamp: string;
}

interface ExportImportProps {
  onExport?: (data: ExportData) => void;
  onImport?: (data: ExportData) => void;
  exportData?: ExportData;
}

/**
 * Export/Import functionality for user data
 * Supports JSON export/import with validation
 */
const ExportImport: React.FC<ExportImportProps> = ({
  onExport,
  onImport,
  exportData
}) => {
  const [importError, setImportError] = useState<string | null>(null);
  const [importSuccess, setImportSuccess] = useState(false);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleExport = () => {
    if (!exportData) return;

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `bubble-export-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    onExport?.(exportData);
  };

  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const data = JSON.parse(text) as ExportData;

        // Validate data structure
        if (!data.sessions || !data.settings || !data.timestamp) {
          throw new Error('Invalid data format');
        }

        setImportError(null);
        setImportSuccess(true);
        onImport?.(data);

        // Reset success message
        setTimeout(() => setImportSuccess(false), 3000);
      } catch (error) {
        setImportError('Failed to import data. Please check the file format.');
        setImportSuccess(false);
      }
    };
    reader.readAsText(file);

    // Reset input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <button
          onClick={handleExport}
          className="bg-cyan-600 hover:bg-cyan-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
          disabled={!exportData}
        >
          Export Data
        </button>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="bg-gray-700 hover:bg-gray-600 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
        >
          Import Data
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          onChange={handleImport}
          className="hidden"
        />
      </div>

      {importError && (
        <div className="bg-red-900/50 border border-red-500/50 rounded-lg p-3 text-red-300 text-sm">
          {importError}
        </div>
      )}

      {importSuccess && (
        <div className="bg-green-900/50 border border-green-500/50 rounded-lg p-3 text-green-300 text-sm">
          Data imported successfully!
        </div>
      )}

      <div className="text-xs text-gray-400">
        Export includes: sessions, settings, and progress data. Import will merge with existing data.
      </div>
    </div>
  );
};

export default ExportImport;

