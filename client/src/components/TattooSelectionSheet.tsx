import React from 'react';
import Sheet from '@mui/joy/Sheet';
import { Close as CloseIcon } from '@mui/icons-material';
import styles from '@/styles/TattooSelectionSheet.module.css';
import cardTattoo from '../images/tattoos/cardTattoo.jpg';
import flowersTattoo from '../images/tattoos/flowersTattoo.jpg';
import swordTattoo from '../images/tattoos/swordTattoo.jpg';
import tribalTattoo from '../images/tattoos/tribalTattoo.jpg';
import skeletonArmTattoo from '../images/tattoos/skeletonArmTattoo.jpg';
import starTattoo from '../../../flask-server/images/star.jpg'
import Image from 'next/image';
import { Grid } from '@mui/material'

interface TattooSelectionProps {
    isOpen: boolean;
    setIsOpen: (bool: boolean) => void;
    setSrc: (str: string) => void;
}

const TattooSelectionSheet = ({ isOpen, setIsOpen, setSrc }: TattooSelectionProps) => {

    const handleTattooClick = (tattoo: string) => {
        console.log(tattoo);
        setSrc(tattoo)
        setIsOpen(false)
    }

    const handleExit = () => {
        setIsOpen(false);
    }

    const height = 130;

    return (
        <>
            {isOpen ? (
                <div className={styles.sheet}>
                    <button onClick={handleExit}>
                        <CloseIcon className={styles.cross} />
                    </button>
                    <Grid
                        container
                        rowSpacing={1}
                        columnSpacing={{ xs: 1, sm: 2, md: 3 }}
                        className={styles.tattooButtons}
                    >
                        <Grid item xs={4}>
                            <button
                                className={styles.tattooButton}
                                onClick={() => handleTattooClick(cardTattoo.src)}
                            >
                                <Image width={height} height={height} src={cardTattoo.src} alt="Card Tattoo" />
                            </button>
                        </Grid>
                        <Grid item xs={4}>
                            <button
                                className={styles.tattooButton}
                                onClick={() => handleTattooClick(starTattoo.src)}
                            >
                                <Image width={height} height={height} src={starTattoo.src} alt="Star Tattoo" />
                            </button>
                        </Grid>
                        <Grid item xs={4}>
                            <button
                                className={styles.tattooButton}
                                onClick={() => handleTattooClick(tribalTattoo.src)}
                            >
                                <Image width={height} height={height} src={tribalTattoo.src} alt="Tribal Tattoo" />
                            </button>
                        </Grid>
                        <Grid item xs={4}>
                            <button
                                className={styles.tattooButton}
                                onClick={() => handleTattooClick(flowersTattoo.src)}
                            >
                                <Image width={height} height={height} src={flowersTattoo.src} alt="Flowers Tattoo" />
                            </button>
                        </Grid>
                        <Grid item xs={4}>
                            <button
                                className={styles.tattooButton}
                                onClick={() => handleTattooClick(swordTattoo.src)}
                            >
                                <Image width={height} height={height} src={swordTattoo.src} alt="Sword Tattoo" />
                            </button>
                        </Grid>
                        <Grid item xs={4}>
                            <button
                                className={styles.tattooButton}
                                onClick={() => handleTattooClick(skeletonArmTattoo.src)}
                            >
                                <Image width={height} height={height} src={skeletonArmTattoo.src} alt="Skeleton Arm Tattoo" />
                            </button>
                        </Grid>
                    </Grid>
                </div>
            ) : null}
        </>
    );
};

export default TattooSelectionSheet;
