// FileInputButton.tsx
import React, { useRef } from 'react';
import emptyFrame from '../images/Empty-frame.png'
import Image from 'next/image';
import styles from '@/styles/FileInputButton.module.css'
interface FileInputButtonProps {
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  text: string;
  image: string | null;
}

const FileInputButton: React.FC<FileInputButtonProps> = ({ onChange, text, image }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className={styles.container} onClick={handleClick}>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={onChange}
      />
      <div className={styles.centered}>{text}</div>
      <Image src={image || emptyFrame.src} alt="empty picture" width='200' height='320' />
    </div>
  );
};

export default FileInputButton;