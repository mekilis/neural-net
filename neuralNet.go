/*
	Created by Smart Mek on Jun 28, 2018 2:53 PM
*/

package main

import (
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
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
func (n *neuralNet) backPropagate(x, y, wHidden, bHidden, wOut, bOut, out *mat.Dense) error {

	return nil
}

func main() {

}
