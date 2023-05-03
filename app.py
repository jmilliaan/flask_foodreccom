## REKOMENDASI MAKANAN
## PROGRAM PENGENALAN DATA SCIENCE YCAB
### Joy Milliaan


from flask import Flask, render_template, request, redirect, url_for
from process import Consumable, empty_df_html
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)
n_input = 3
n_output = 3
makanan_loc = "dataset/makanan/data_makanan.csv"
minuman_loc = "dataset/minuman/data_minuman.csv"
cemilan_loc = "dataset/cemilan/data_cemilan.csv"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/makanan", methods=["GET", "POST"])
def makanan():
    makanan = Consumable(makanan_loc, n_input, n_output)
    options = makanan.generate_options_remaining()
    print("options", options)
    opt1, opt2, opt3 = options.iloc[0]["Nama"], options.iloc[1]["Nama"], options.iloc[2]["Nama"]
    print("opts", opt1, opt2, opt3)
    if request.method == "POST":
        action = request.form["action"]
        post_data = action.split(";")
        
        if post_data[0] == "show_table":
            m1 = request.form.get("makanan_1")
            print(f"m1: {m1}")
            slider1 = int(request.form["radio1"])
            slider2 = int(request.form["radio2"])
            slider3 = int(request.form["radio3"])
            
            ratings = [slider1, slider2, slider3]
            makanan.get_user_ratings(ratings)
            makanan.generate_rating_vector()
            makanan.generate_similarity()
            
            final = makanan.recommend().drop("Score", axis=1).reset_index(drop=True)
            table = final.to_html(classes="table table-striped", index=False)
            return render_template(
                "makanan.html",
                makanan_1=post_data[1], 
                makanan_2=post_data[2], 
                makanan_3=post_data[3],
                slider1=slider1, 
                slider2=slider2, 
                slider3=slider3, 
                table=table,
                value_1=slider1, 
                value_2=slider2, 
                value_3=slider3)
        
    return render_template(
        "makanan.html", 
        makanan_1=opt1, 
        makanan_2=opt2, 
        makanan_3=opt3,
        value_1=1, 
        value_2=1, 
        value_3=1)

@app.route("/minuman", methods=["GET", "POST"])
def minuman():
    minuman = Consumable(minuman_loc, n_input, n_output)
    options = minuman.generate_options_remaining()
    opt1, opt2, opt3 = options.iloc[0]["Nama"], options.iloc[1]["Nama"], options.iloc[2]["Nama"]
    if request.method == "POST":
        action = request.form["action"]
        post_data = action.split(";")
        if post_data[0] == "show_table":
            slider1 = int(request.form["radio1"])
            slider2 = int(request.form["radio2"])
            slider3 = int(request.form["radio3"])
            ratings = [slider1, slider2, slider3]
            minuman.get_user_ratings(ratings)
            minuman.generate_rating_vector()
            minuman.generate_similarity()
            final = minuman.recommend().drop("Score", axis=1).reset_index(drop=True)
            table = final.to_html(classes="table table-striped", index=False)

            return render_template(
                "minuman.html",
                minuman_1=post_data[1], 
                minuman_2=post_data[2], 
                minuman_3=post_data[3],
                slider1=slider1, 
                slider2=slider2, 
                slider3=slider3, 
                table=table,
                value_1=slider1, 
                value_2=slider2, 
                value_3=slider3)
        
    return render_template(
        "minuman.html", 
        minuman_1=opt1, 
        minuman_2=opt2, 
        minuman_3=opt3,
        value_1=1, 
        value_2=1, 
        value_3=1)

@app.route("/cemilan", methods=["GET", "POST"])
def cemilan():
    cemilan = Consumable(cemilan_loc, n_input, n_output)
    options = cemilan.generate_options_remaining()
    opt1, opt2, opt3 = options.iloc[0]["Nama"], options.iloc[1]["Nama"], options.iloc[2]["Nama"]
    if request.method == "POST":
        action = request.form["action"]
        post_data = action.split(";")
        if post_data[0] == "show_table":
            slider1 = int(request.form["radio1"])
            slider2 = int(request.form["radio2"])
            slider3 = int(request.form["radio3"])
            ratings = [slider1, slider2, slider3]
            cemilan.get_user_ratings(ratings)
            cemilan.generate_rating_vector()
            cemilan.generate_similarity()
            final = cemilan.recommend().drop("Score", axis=1).reset_index(drop=True)
            table = final.to_html(classes="table table-striped", index=False)
            return render_template(
                "cemilan.html",
                cemilan_1=post_data[1], 
                cemilan_2=post_data[2], 
                cemilan_3=post_data[3],
                slider1=slider1, 
                slider2=slider2, 
                slider3=slider3, 
                table=table,
                value_1=slider1, 
                value_2=slider2, 
                value_3=slider3)
    
    return render_template(
        "cemilan.html", 
        cemilan_1=opt1, 
        cemilan_2=opt2, 
        cemilan_3=opt3,
        value_1=1, 
        value_2=1, 
        value_3=1)
@app.route('/reset', methods=['POST'])
def reset():
    return redirect(url_for(
        'reset', 
        slider1=1, 
        slider2=1, 
        slider3=1, 
        table=empty_df_html
        )
    )


if __name__ == '__main__':
    app.run(debug=True)
