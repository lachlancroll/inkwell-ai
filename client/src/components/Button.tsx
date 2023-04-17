import styles from '@/styles/Button.module.css';

interface ButtonProps {
  onClick: () => void;
  text: string;
}

const Button = ({ onClick, text }: ButtonProps) => {
  return (
    <div className={styles.wrapper}>
      <button className={styles.button} onClick={onClick}>
        {text}
      </button>
    </div>
  );
};

export default Button;