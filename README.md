# nishantkelkar-blog.github.io

Github Pages blog site for random musings in my life

## Running the code

- If a new computer installation, run `gem install --user-install bundler jekyll`

- Start a node + jekyll server: `bundle exec jekyll serve`

- Validate on the printed `localhost` address that your blog server is up.

## Adding a new blog entry

- Create a new file in the format `<YYYY-MM-dd>_<name>.md` in `_posts/`.

- Add appropriate preamble/front-matter material to this file.
  An example preamble is like the following:

      ```text
      ---
      layout: post
      author: Nishant Kelkar
      title: Vector calculus for deep learning
      tags: machine-learning deep-learning
      ---
      ```

- To generate tags so that they show up on the blog, now run `python tag_generator.py` from the root directory.

- Push to Github and validate your new blog entry within a few minutes!

- GitHub help page for running Jekyll sites: https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll#keeping-your-site-up-to-date-with-the-github-pages-gem
