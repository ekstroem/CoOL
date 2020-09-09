// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
using namespace Rcpp;


/****
   Andreas Rieckmann
   Piotr Dworzynski
   Claus Ekstrøm
   2019
****/

//' Function used as part of other functions
//'
//'
//' @description relu-function
//' @param x input in the relu function
//'
//' @export
// [[Rcpp::export]]
arma::mat rcpprelu(const arma::mat & x) {
    arma::mat m = x%(x>0);
    return m;
}

//' Function used as part of other functions
//'
//'
//' @description negative relu-function
//' @param x input in the negative relu-function
//'
//' @export
// [[Rcpp::export]]
arma::mat rcpprelu_neg(const arma::mat & x) {
    arma::mat m = x%(x<0);
    return m;
}

//' Function used as part of other functions
//'
//'
//' @description Used as part of other functions.
//' @param x A matrix of predictors for the training dataset
//' @param y A vector of output values for the training data with a length similar to the number of rows of x
//' @param testx A matrix of predictors for the test dataset
//' @param testy A vector of output values for the test data with a length similar to the number of rows of x
//' @param W1_input Input-hidden layer weights
//' @param B1_input Biases for the hidden layer
//' @param W2_input Hidden-output layer weights
//' @param B2_input Bias for the output layer (the baseline risk)
//' @param lr Initial learning rate
//' @param maxepochs The maximum number of epochs
//' @param baseline_risk_reward increasing parameter value at each iteration
//' @param IPCW Inverse probability of censoring weights (Warning: not yet correctly implemented)
//' @return A list of class "SCL" giving the estimated matrices and performance indicators
//' @author    Andreas Rieckmann, Piotr Dworzynski, Claus Ekstrøm
//'
//' @export
// [[Rcpp::export]]
Rcpp::List CoOL_cpp_train_network_relu(
     const arma::mat & x,
		 const arma::vec & y,
     const arma::mat & testx,
     const arma::vec & testy,
     const arma::mat & W1_input,
     const arma::mat & B1_input,
     const arma::mat & W2_input,
     const arma::mat & B2_input,
     const arma::vec & IPCW,
		 double lr=0.01,
		 double maxepochs = 100,
     double L1 = 0.00001
		 ) {

  int nsamples = y.size();
  int nfeatures = x.n_cols;
  int hidden = W1_input.n_cols;
  int sparse_data = 0;
  double mean_y = accu(y) / nsamples;

  Rprintf("%s \n", "CoOL");

  // Loaded initialized weights.
  arma::mat W1(nfeatures, hidden, arma::fill::zeros);  // Filled with standard normals
  W1 = W1_input;
  arma::mat B1(1, hidden, arma::fill::zeros);
  B1 = B1_input;
  arma::mat W2(hidden, 1, arma::fill::zeros);  // Filled with standard normals
  W2 = W2_input;
  arma::mat B2(1, 1, arma::fill::zeros);
  B2 = B2_input;

  // W1 for the test data parameter qualification
  arma::mat W1_previous_step(nfeatures, hidden, arma::fill::zeros);  // Filled with standard normals
  W1_previous_step = W1_input;
  arma::mat W1_temp(nfeatures, hidden, arma::fill::zeros);  // Filled with standard normals
  W1_temp = W1_input;
  arma::mat W1_next_step(nfeatures, hidden, arma::fill::zeros);  // Filled with standard normals
  W1_next_step = W1_input;


  // Define temporary holders and predicted output
  arma::mat h(1, hidden);
  arma::vec o(nsamples);
 
  arma::vec trainperf(maxepochs, arma::fill::zeros);
  arma::vec testperf(maxepochs, arma::fill::zeros);

  trainperf.replace(0, arma::datum::nan);
  testperf.replace(0, arma::datum::nan);

 // monitor the difference in weights
  arma::vec trainweights(maxepochs, arma::fill::zeros);
  trainweights.replace(0, arma::datum::nan);

   // monitor the baseline risk
  arma::vec baseline_risks(maxepochs, arma::fill::zeros);
  baseline_risks.replace(0, arma::datum::nan);

  arma::vec index = arma::linspace<arma::vec>(0, nsamples-1, nsamples);

  int row;
  // Main loop

  arma::uword epoch=0;
  for (epoch=0; epoch < maxepochs; epoch++) {
    // First we shuffle/permute all row indices before commencing
    arma::vec shuffle = arma::shuffle(index);
    //w = epoch / maxepochs;
    // Decay the learning rate
    //lr = std::max(decay*lr, 0.00001);

    //    Rcout << "B1 at start of epoch "<< epoch << " is " << B1 << std::endl;

    // Step 2: Implement forward propagation to get hW,b(x) for any x.

    for (arma::uword rowidx=0; rowidx<x.n_rows; rowidx++) {
      // This is the row we're working on right now
      row = shuffle(rowidx);
      //      row = rowidx;

      // h contains the output from the hidden layer.
      // h has dimension 1 x hidden
      // it is a vector for individual "row" with hidden elements
      h = rcpprelu((x.row(row) * (W1)) + B1);

      // Now do the same to get the output layer
      o(row) = rcpprelu(h * W2 + B2)(0,0); // the relu function is redundant

      // Okay ... Forward done ... Now we need to back propagate
      double E_outO = - (y(row) - o(row));
       arma::mat netO_wHO = trans(h);
      //arma::mat outH_netH = h>0; 
      arma::mat netO_outH = trans(W2);
    // Possibly move this to after the trans line below
    // W2 = rcpprelu(W2 - lr*E_outO * netO_wHO);

      // All calculations done. Now do the updating
      for (size_t g=0; g<W1.n_rows; g++) {
        W1.row(g) = rcpprelu(W1.row(g) - IPCW(row) * lr * E_outO * (netO_outH % (h>0)) * x(row, g)); // - lr * L1); // L1 regularized - penalized
}

      B1 = rcpprelu_neg(B1 - IPCW(row) * lr * E_outO * (netO_outH % (h>0)));
      B2 = rcpprelu(B2 - IPCW(row) * lr / 10 *  E_outO + lr * L1); // inverse L1 regularized - rewarded


    } // Row

/*
    // Parameter qualification based on test data performance
      for (size_t g=0; g<W1.n_rows; g++) {
         for (size_t j=0; j<W1.n_cols; j++) {
        W1_temp = W1_previous_step;
        W1_temp(g,j) = W1(g,j);
        arma::mat tmp = testx * W1_previous_step;
        double mean_val_perform_prev = 0.5*accu(square(testy - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2 + B2(0,0)) )))/nsamples;
        tmp = testx * W1_temp;
        double mean_val_perform_next = 0.5*accu(square(testy - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2 + B2(0,0)) )))/nsamples;
          if (mean_val_perform_next <= mean_val_perform_prev) {
            W1_next_step(g,j) = W1(g,j);
          }
          if (mean_val_perform_next > mean_val_perform_prev) {
            W1_next_step(g,j) = W1_previous_step(g,j) * 0.9;
          }
        }
      }
    W1 = W1_next_step;
    W1_previous_step = W1_next_step;
*/
    
    // Compute performance
    // This is an approximation since it should be rerun on the full data with the
    // new weights.  But we have an update of a single line
    arma::mat tmp = x * W1;
    double mean_perform = 0.5*accu(square(y - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2 + B2(0,0)))))/nsamples;

    // Compute performance on the validation (test) set    
    tmp = testx * W1;
    double mean_val_perform = 0.5*accu(square(testy - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2 + B2(0,0)) )))/nsamples;

    trainperf(epoch) = mean_perform;
    testperf(epoch) = mean_val_perform;

    // Calculating the mean squared difference in weight update
    double mean_w_diff = 0;
   for (size_t i=0; i<W1.n_rows; i++) {
      for (size_t g=0; g<W1.n_cols; g++) {
    mean_w_diff = mean_w_diff + (W1(i,g) - W1_previous_step(i,g)) * (W1(i,g) - W1_previous_step(i,g));
    }
  }
    mean_w_diff = mean_w_diff / (W1.n_rows * W1.n_cols);
    //Rprintf("%f",mean_w_diff);
    trainweights(epoch) = mean_w_diff;
    W1_previous_step = W1;

    // Monitor the baseline risk
    baseline_risks(epoch) = B2(0,0);

  if (B2(0,0) == 0) {
    sparse_data = 1;
  }

 if (epoch % 10 == 0) {
 //     Rprintf("%d epochs: Train performance of %f. Test performance of %f. Baseline risk estimated to %f.\n",epoch, mean_perform, mean_val_perform, B2(0,0));
      Rprintf("%d epochs: Train performance of %f. Baseline risk estimated to %f.\n",epoch, mean_perform, B2(0,0));
  // Warnings:
  if (B2(0,0) > mean_y) {
  Rprintf("Warning: The baseline risk (%f) is higher than mean(Y) (%f)! Consider reducing the regularisation of the baseline risk.\n", B2(0,0), mean_y);
   }
  if (sparse_data == 1) {
  Rprintf("Warning: The baseline risk (%f) has at one time been estimated to zero. Data may be too sparse.\n", B2(0,0));
   }
   }



  }  // End of all epochs
  arma::colvec trainp = trainperf.elem(arma::find_finite(trainperf));
  arma::colvec testp = testperf.elem(arma::find_finite(testperf));
  arma::colvec trainp_weights = trainweights.elem(arma::find_finite(trainweights));
  arma::colvec baseline_risks_monitor = baseline_risks.elem(arma::find_finite(baseline_risks));


  Rcpp::List RVAL =  Rcpp::List::create(
          Rcpp::Named("W1")=W1,
          Rcpp::Named("B1")=B1,
          Rcpp::Named("W2")=W2,
          Rcpp::Named("B2")=B2,
          Rcpp::Named("train_performance")=trainp,
          Rcpp::Named("test_performance")=testp,
          Rcpp::Named("weight_performance")=trainp_weights,
          Rcpp::Named("baseline_risk_monitor")=baseline_risks_monitor,
          Rcpp::Named("epochs")=epoch+1
                          );

  RVAL.attr("class") = CharacterVector::create("SCL", "list");

  return(RVAL);
}






//' Function used as part of other functions
//'
//'
//' @description Used as part of other functions.
//' @param x A matrix of predictors for the training dataset
//' @param y A vector of output values for the training data with a length similar to the number of rows of x
//' @param c A matrix of predictors for the training data to be regarded as potential confounder(s)
//' @param testx A matrix of predictors for the test dataset
//' @param testy A vector of output values for the test data with a length similar to the number of rows of x
//' @param testc A matrix of predictors for the test data to be regarded as potential confounder(s)
//' @param W1_input Input-hidden layer weights
//' @param B1_input Biases for the hidden layer
//' @param W2_input Hidden-output layer weights
//' @param B2_input Bias for the output layer (the baseline risk)
//' @param C2_input Weight for the confounder
//' @param lr Initial learning rate
//' @param maxepochs The maximum number of epochs
//' @return A list of class "SCL" giving the estimated matrices and performance indicators
//' @author    Andreas Rieckmann, Piotr Dworzynski, Claus Ekstrøm
//'
//' @export
// [[Rcpp::export]]
Rcpp::List CoOL_cpp_train_network_relu_with_confounder(
     const arma::mat & x,
     const arma::vec & y,
     const arma::mat & c,
     const arma::mat & testx,
     const arma::vec & testy,
     const arma::mat & testc,
     const arma::mat & W1_input,
     const arma::mat & B1_input,
     const arma::mat & W2_input,
     const arma::mat & B2_input,
     const arma::mat & C2_input,
     double lr=0.01,
     double maxepochs = 100
     ) {

  int nsamples = y.size();
  int nfeatures = x.n_cols;
  int hidden = W1_input.n_cols;


  Rprintf("%s \n", "CoOL");

  // Loaded initialized weights.
  arma::mat W1(nfeatures, hidden, arma::fill::zeros);  // Filled with standard normals
  W1 = W1_input;
  arma::mat B1(1, hidden, arma::fill::zeros);
  B1 = B1_input;
  arma::mat W2(hidden, 1, arma::fill::zeros);  // Filled with standard normals
  W2 = W2_input;
  arma::mat B2(1, 1, arma::fill::zeros);
  B2 = B2_input;

  arma::mat C2(1, 1, arma::fill::zeros);
  C2 = C2_input;

  // Define temporary holders and predicted output
  arma::mat h(1, hidden);
  arma::vec o(nsamples);
 
  arma::vec trainperf(maxepochs, arma::fill::zeros);
  arma::vec testperf(maxepochs, arma::fill::zeros);

  trainperf.replace(0, arma::datum::nan);
  testperf.replace(0, arma::datum::nan);

  arma::vec index = arma::linspace<arma::vec>(0, nsamples-1, nsamples);

  int row;
  // Main loop

  arma::uword epoch=0;
  for (epoch=0; epoch < maxepochs; epoch++) {
    // First we shuffle/permute all row indices before commencing
    arma::vec shuffle = arma::shuffle(index);
    //w = epoch / maxepochs;
    // Decay the learning rate
    //lr = std::max(decay*lr, 0.00001);

    //    Rcout << "B1 at start of epoch "<< epoch << " is " << B1 << std::endl;

    // Step 2: Implement forward propagation to get hW,b(x) for any x.
    for (arma::uword rowidx=0; rowidx<x.n_rows; rowidx++) {
      // This is the row we're working on right now
      row = shuffle(rowidx);
      //      row = rowidx;

      // h contains the output from the hidden layer.
      // h has dimension 1 x hidden
      // it is a vector for individual "row" with hidden elements
      h = rcpprelu((x.row(row) * (W1)) + B1);

      // Now do the same to get the output layer
      o(row) = rcpprelu(h * W2 + B2 + c.row(row)*C2)(0,0); // the relu function is redundant

      // Okay ... Forward done ... Now we need to back propagate
      double E_outO = - (y(row) - o(row));
      arma::mat netO_wHO = trans(h);
      //arma::mat outH_netH = h>0; 
      arma::mat netO_outH = trans(W2);
    // Possibly move this to after the trans line below
    //  W2 = rcpprelu(W2 - lr*E_outO * netO_wHO);

      // All calculations done. Now do the updating
      for (size_t g=0; g<W1.n_rows; g++) {
        W1.row(g) = rcpprelu(W1.row(g) - 10* lr * E_outO *  (netO_outH % (h>0)) * x(row, g));
}

      B1 = rcpprelu_neg(B1 - lr * E_outO *  (netO_outH % (h>0)));
      B2 = rcpprelu(B2 -lr * 0.1 * E_outO );

      // Update confounder weight
      netO_wHO = trans(c.row(row));
      C2 -= lr * E_outO * netO_wHO;

    } // Row

    // Compute performance
    // This is an approximation since it should be rerun on the full data with the
    // new weights.  But we have an update of a single line
    arma::mat tmp = x * W1;
    double mean_perform = 0.5*accu(square(y - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2 + B2(0,0) + c*C2))))/nsamples;

    // Compute performance on the validation (test) set    
    tmp = testx * W1;
    double mean_val_perform = 0.5*accu(square(testy - (rcpprelu(rcpprelu(tmp.each_row() + B1) * W2 + B2(0,0) + testc*C2))))/nsamples;

    testperf(epoch) = mean_val_perform;
    trainperf(epoch) = mean_perform;

 if (epoch % 10 == 0) {
      Rprintf("%d epochs: Test performance of %f. Baseline risk estimated to %f.\n",epoch, mean_perform, B2(0,0));
   }



  }  // End of all epochs

  arma::colvec testp = testperf.elem(arma::find_finite(testperf));
  arma::colvec trainp = trainperf.elem(arma::find_finite(trainperf));


  Rcpp::List RVAL =  Rcpp::List::create(
          Rcpp::Named("W1")=W1,
          Rcpp::Named("B1")=B1,
          Rcpp::Named("W2")=W2,
          Rcpp::Named("B2")=B2,
          Rcpp::Named("C2")=C2,
          Rcpp::Named("train_performance")=trainp,
          Rcpp::Named("test_performance")=testp,
          Rcpp::Named("epochs")=epoch+1
                          );

  RVAL.attr("class") = CharacterVector::create("SCL", "list");

  return(RVAL);
  
  
}





/*




//////// DEBUG ////////////
  Rcpp::List RVAL =  Rcpp::List::create(
          Rcpp::Named("W1")= W1,
          Rcpp::Named("B1")= B1,
          Rcpp::Named("W2")= W2,
          Rcpp::Named("B2")= B2,
          Rcpp::Named("C2")= C2
                          );
  RVAL.attr("class") = CharacterVector::create("SCL", "list");
  return(RVAL);
//////// DEBUG ////////////


//////// DEBUG ////////////
  Rcpp::List RVAL =  Rcpp::List::create(
          Rcpp::Named("W1")= W1
                          );
  RVAL.attr("class") = CharacterVector::create("SCL", "list");
  return(RVAL);
//////// DEBUG ////////////


//////// DEBUG ////////////
  Rcpp::List RVAL =  Rcpp::List::create(
          Rcpp::Named("0.4 * exp(-1*(2*lambda(g) * lambda(g))/2)")= 0.4 * exp(-1*(2*lambda(g) * lambda(g))/2),
          Rcpp::Named("lr")=lr,
          Rcpp::Named("E_outO")=E_outO,
          Rcpp::Named("outO_netO")=outO_netO,         
          Rcpp::Named("netO_outH")=netO_outH,
          Rcpp::Named("outH_netH")=outH_netH,
          Rcpp::Named("exp(W1.row(g))")=exp(W1.row(g)),
          Rcpp::Named("xandc(row, g)")=xandc(row, g),
          Rcpp::Named("0.4 * exp(-1*(2*lambda(g) * lambda(g))/2) * accu(lr * E_outO * outO_netO * (netO_outH _ outH_netH _ exp(W1.row(g))) * xandc(row, g))")=0.4 * exp(-1*(2*lambda(g) * lambda(g))/2) * accu(lr * E_outO * outO_netO * (netO_outH % outH_netH % exp(W1.row(g))) * xandc(row, g))
                          );
  RVAL.attr("class") = CharacterVector::create("SCL", "list");
  return(RVAL);
//////// DEBUG ////////////
*/