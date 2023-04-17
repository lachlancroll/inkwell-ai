// FileInputButton.tsx
import React, { useRef } from 'react';
import styles from '@/styles/FileInputButton.module.css'
import Button from './Button';

interface FileInputButtonProps {
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  text: string;
}

const FileInputButton: React.FC<FileInputButtonProps> = ({ onChange, text }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="file-input-wrapper">
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={onChange}
      />
      <Button onClick={handleClick} text={text} />
    </div>
  );
};

export default FileInputButton;