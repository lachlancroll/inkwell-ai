import Head from 'next/head'
import styles from '@/styles/Home.module.css'
import React, { useState } from 'react';
import logo from '../images/inkwell-logo.png'
import Image from 'next/image';
import Button from '@/components/Button';
import FileInputButton from '@/components/FileInputButton';
import TattooSelection from '@/components/TattooSelection';
import star from '../../../flask-server/images/star.jpg';
import TattooSelectionSheet from '@/components/TattooSelectionSheet';
import { Grid } from '@mui/material'
import emptyFrame from '../images/Empty-frame.png'

const Home = () => {
  const [data, setData] = useState<string | null>(null);
  const [armUpload, setArmUpload] = useState<File | null>(null);
  const [armImage, setArmImage] = useState<string | null>(null);
  const [tattooImage, setTattooImage] = useState<string>(star.src);
  const [x, setX] = useState<number | null>(null);
  const [y, setY] = useState<number | null>(null);
  const [tattooWidth, setTattooWidth] = useState<number | null>(90);
  const [tattooLength, setTattooLength] = useState<number | null>(90);
  const [isOpen, setIsOpen] = useState<boolean>(false);

  const handleArmInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setArmUpload(event.target.files[0]);
      // Create a blob URL for the uploaded file
      const imageUrl = URL.createObjectURL(event.target.files[0]);
      setArmImage(imageUrl);
    }
  };

  const handleGenerate = async () => {
    if (!armUpload || !tattooImage) {
      return;
    }

    const formData = new FormData();
    formData.append('arm_file', armUpload);
    formData.append('x', x ? x.toString() : '');
    formData.append('y', y ? y.toString() : '');
    formData.append('height', tattooWidth ? tattooWidth.toString() : '');
    formData.append('width', tattooLength ? tattooLength.toString() : '');

    // Convert tattoo image to blob
    const tattooImageBlob = await fetch(tattooImage).then(response => response.blob());
    formData.append('tattoo_file', tattooImageBlob);

    fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    })
      .then(response => response.json())
      .then(data => {
        const img_src = `data:image/png;base64,${data.img}`;
        setData(img_src);
      })
      .catch(error => {
        console.error(error);
      });
  }

  const handleChangeTattoo = () => {
    setIsOpen(true);
  }

  return (
    <>
      <Head>
        <title>Create Next App</title>
        <meta name="description" content="Generated by create next app" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className={styles.main}>
        <div className={styles.wrapper}>
          <Image src={logo.src} alt="inkwell logo" width='200' height='140' />
        </div>
        <Grid
          container
          rowSpacing={1}
          columnSpacing={{ xs: 1, sm: 2, md: 3 }}
          className={styles.tattooButtons}
        >
          <Grid item xs={4}>
            <div className={styles.fileUpload}>
              <FileInputButton image={armImage} onChange={handleArmInputChange} text='Upload Arm Image' />
            </div>
          </Grid>
          <Grid item xs={4}>
            <div className={styles.fileUpload}>
              <TattooSelection setY={setY} setX={setX} src={tattooImage} />
            </div>
          </Grid>
          <Grid item xs={4}>
            <div className={styles.fileUpload}>
              <Image src={data ? data : emptyFrame.src} alt="random image" width='200' height='320' />
            </div>
          </Grid>
          <Grid item xs={4}>
            <Button onClick={() => null}>
              Upload arm image
            </Button>
          </Grid>
          <Grid item xs={4}>
            <Button onClick={handleChangeTattoo}>
              Change Tattoo
            </Button>
          </Grid>
          <Grid item xs={4}>
            <Button onClick={handleGenerate}>
              Generate
            </Button>
          </Grid>
        </Grid>
      </main>
      <TattooSelectionSheet isOpen={isOpen} setIsOpen={setIsOpen} setSrc={setTattooImage} />
    </>
  )
}

export default Home;