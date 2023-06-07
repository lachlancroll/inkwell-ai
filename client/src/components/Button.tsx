import styles from '@/styles/Button.module.css';

interface ButtonProps {
  onClick: () => void;
  children: React.ReactNode;
}

const Button = ({ onClick, children }: ButtonProps) => {
  return (
    <div className={styles.container}>
      <button className={styles.button} onClick={onClick}>
        {children}
      </button>
    </div>
  );
};

export default Button;