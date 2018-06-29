/*
	Created by Smart Mek on Jun 28, 2018 2:53 PM
*/

package main

import (
	"math"
	"math/rand"
	"time"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var (
	ERR_INVALID_AXIS = errors.New("invalid axis. axis not zero or one")
)

type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

func NewNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{
		config: config,
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return x * (1.0 - x)
}

func (n *neuralNet) train(x, y *mat.Dense) error {
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	wHidden := mat.NewDense(n.config.inputNeurons, n.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, n.config.hiddenNeurons, nil)

	wOut := mat.NewDense(n.config.hiddenNeurons, n.config.outputNeurons, nil)
	bOut := mat.NewDense(1, n.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	params := [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	}

	for _, param := range params {
		for p := range param {
			param[p] = r.Float64()
		}
	}

	out := new(mat.Dense)

	err := n.backPropagate(x, y, wHidden, bHidden, wOut, bOut, out)
	if err != nil {
		return err
	}

	n.wHidden = wHidden
	n.bHidden = bHidden
	n.wOut = wOut
	n.bOut = bOut

	return nil
}
func (n *neuralNet) backPropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	// 'Apply' functions

	addBHidden := func(_, col int, v float64) float64 {
		return v + bHidden.At(0, col)
	}

	addBOut := func(_, col int, v float64) float64 {
		return v + bOut.At(0, col)
	}

	applySigmoid := func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}

	applySigmoidPrime := func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}

	for i := 0; i < n.config.numEpochs; i++ {

		hiddenLayerInput := new(mat.Dense)
		hiddenLayerActivations := new(mat.Dense)
		slopeHiddenLayer := new(mat.Dense)
		errAtHiddenLayer := new(mat.Dense)
		dHiddenLayer := new(mat.Dense)

		networkError := new(mat.Dense)

		outputLayerInput := new(mat.Dense)
		slopeOutputLayer := new(mat.Dense)
		dOutput := new(mat.Dense)

		/*

							<----		hidden layer	---><--	output layer	------>

										_________						back propagate errors o1 and o2
										|		|			 _______________________
										|		|\ e1		/	_________			|
										|		| \	\	   /	|		|			|
					------------------->|		|  \	\ /	w11	|		|			|
										|		|	\		\	|		|	o1		|
										|		|	 \		   \|		|___________|_______\	<output1>
										|_______|	  \	   w21//|		|					/
													   \	//	|		|
										_________		/	/	|		|
										|		|	/	 \ /w31	|_______|
										|		|/ e2 	  /		_________
										|		|	\	 / \w12	|		|
					------------------->|		|	   \/	\	|		|
										|		|	   /	\\	|		|	o2
										|		|	  /	   w22\\|		|___________________\	<output2>
										|_______|	 /		   /|		|			|		/
													/	  w32/	|		|			|
										_________  /	/		|		|			|
										|		| /	/		/|\	|_______|			|
										|		|/ e3		 |______________________|
										|		|
					------------------->|		|
										|		|
										|		|
										|_______|

						-			-						-			 			  -			-		-
						| w11	w12	|			-	 -		| (w11 x o1) + (w12 x o2) |			|	e1	|
						| w21   w22	|	  *		| o1 |  =	| (w21 x o1) + (w22 x o2) |		= 	|	e2	|
						| w31	w32	|			| o2 | 		| (w21 x o1) + (w32 x o2) |			|	e3	|
						-			-			- 	 -		-			 			  -			-		-

					 --> Transpose of <-- 	---> error <---									---> error from <---
						weight to output	from output layer									hidden layer
							layer

							wOut^T				dOutput										  errAtHiddenLayer

		*/

		hiddenLayerInput.Mul(x, wHidden)
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		networkError.Sub(y, output)
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput.MulElem(networkError, slopeOutputLayer)

		errAtHiddenLayer.Mul(dOutput, wOut.T())
		dHiddenLayer.MulElem(errAtHiddenLayer, slopeHiddenLayer)

		// Adjust weight and bias
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(n.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(n.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.MulElem(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(n.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}

		bHiddenAdj.Scale(n.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)

	}

	return nil
}

func sumAlongAxis(axis int, a *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := a.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, a)
			data[i] = floats.Sum(col)
		}

		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, 1, a)
			data[i] = floats.Sum(row)
		}

		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, ERR_INVALID_AXIS
	}

	return output, nil

}

func main() {

}
