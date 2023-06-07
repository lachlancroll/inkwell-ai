import React, { useState, useEffect } from 'react';
import { Stage, Layer, Image as KonvaImage } from 'react-konva';
import arm from '../images/arm.png'

interface TattooSelection {
  setY: (num: number) => void;
  setX: (num: number) => void;
  src: string;
}

const TattooSelection = ({ setX, setY, src }: TattooSelection) => {
  const [image, setImage] = useState<CanvasImageSource | undefined>(undefined);
  const [draggableImage, setDraggableImage] = useState<CanvasImageSource | undefined>(undefined);

  useEffect(() => {
    const imageObj = new window.Image();
    imageObj.src = src;
    imageObj.onload = () => {
      // Create an off-screen canvas
      const canvas = document.createElement('canvas');
      canvas.width = imageObj.width;
      canvas.height = imageObj.height;

      // Get a 2D rendering context
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(imageObj, 0, 0);

        // Get the image data
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        // Loop through all pixels
        for (let i = 0; i < data.length; i += 4) {
          // Change all white (also shades of whites)
          // pixels to be transparent
          if (data[i] > 200 && data[i + 1] > 200 && data[i + 2] > 200)
            data[i + 3] = 0;
        }

        ctx.putImageData(imageData, 0, 0);

        // Create a new image with the modified data
        const newImage = new window.Image();
        newImage.src = canvas.toDataURL();
        setDraggableImage(newImage);
      }
    }
  }, [src]);



  useEffect(() => {
    const imageObj = new window.Image();
    imageObj.src = arm.src;
    imageObj.onload = () => {
      setImage(imageObj);
    }
  }, []);

  const handleDragEnd = (e: any) => {
    console.log(e.target.attrs)
    let newX = e.target.attrs.x;
    let newY = e.target.attrs.y;
    if (newX > 60) {
      newX = 60;
    } else if (newX < 40) {
      newX = 40;
    }

    if (newY > 250) {
      newY = 250;
    } else if (newY < 95) {
      newY = 95;
    }

    setY(newY)
    setX(newX)
    e.target.position({
      x: newX,
      y: newY,
    });
  };

  return (
    <Stage width={180} height={320} onClick={() => null}>
      <Layer>
        <KonvaImage image={image} width={180} height={320} />
        <KonvaImage
          image={draggableImage}
          width={90}
          height={90}
          draggable
          onDragEnd={handleDragEnd}
        />
      </Layer>
    </Stage>
  )
}

export default TattooSelection;